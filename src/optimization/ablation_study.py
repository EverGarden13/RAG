"""
Ablation study framework for analyzing component contributions.
Tests impact of removing or modifying individual components.
"""

import json
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AblationResult:
    """Result from an ablation experiment."""
    experiment_name: str
    description: str
    em_score: float
    ndcg_score: float
    combined_score: float
    components_removed: List[str] = field(default_factory=list)
    components_modified: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AblationStudyFramework:
    """Framework for conducting ablation studies."""
    
    def __init__(self, baseline_fn: Callable,
                 validation_data: List[Dict[str, Any]],
                 output_dir: str = "./ablation_results"):
        """
        Initialize ablation study framework.
        
        Args:
            baseline_fn: Function that evaluates baseline system
            validation_data: Validation dataset
            output_dir: Directory to save results
        """
        self.baseline_fn = baseline_fn
        self.validation_data = validation_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[AblationResult] = []
        self.baseline_result: Optional[AblationResult] = None
        
        logger.info(f"Ablation study framework initialized with {len(validation_data)} validation samples")
    
    def run_baseline(self) -> AblationResult:
        """
        Run baseline system with all components.
        
        Returns:
            Baseline ablation result
        """
        logger.info("\n" + "="*60)
        logger.info("RUNNING BASELINE SYSTEM")
        logger.info("="*60)
        
        try:
            em, ndcg, combined = self.baseline_fn({})
            
            self.baseline_result = AblationResult(
                experiment_name="baseline",
                description="Full system with all components enabled",
                em_score=em,
                ndcg_score=ndcg,
                combined_score=combined,
                metadata={'is_baseline': True}
            )
            
            self.results.append(self.baseline_result)
            
            logger.info(f"Baseline Results:")
            logger.info(f"  EM: {em:.4f}")
            logger.info(f"  nDCG: {ndcg:.4f}")
            logger.info(f"  Combined: {combined:.4f}")
            logger.info("="*60 + "\n")
            
            return self.baseline_result
            
        except Exception as e:
            logger.error(f"Error running baseline: {e}")
            raise
    
    def ablate_component(self, component_name: str,
                        evaluation_fn: Callable,
                        description: str = "") -> AblationResult:
        """
        Remove a component and evaluate impact.
        
        Args:
            component_name: Name of component to remove
            evaluation_fn: Function to evaluate without component
            description: Description of the ablation
            
        Returns:
            Ablation result
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATING: {component_name}")
        logger.info(f"Description: {description}")
        logger.info(f"{'='*60}")
        
        try:
            em, ndcg, combined = evaluation_fn(self.validation_data)
            
            result = AblationResult(
                experiment_name=f"ablate_{component_name}",
                description=description or f"System without {component_name}",
                em_score=em,
                ndcg_score=ndcg,
                combined_score=combined,
                components_removed=[component_name]
            )
            
            self.results.append(result)
            
            # Calculate impact
            if self.baseline_result:
                em_delta = em - self.baseline_result.em_score
                ndcg_delta = ndcg - self.baseline_result.ndcg_score
                combined_delta = combined - self.baseline_result.combined_score
                
                logger.info(f"Results:")
                logger.info(f"  EM: {em:.4f} (Δ {em_delta:+.4f})")
                logger.info(f"  nDCG: {ndcg:.4f} (Δ {ndcg_delta:+.4f})")
                logger.info(f"  Combined: {combined:.4f} (Δ {combined_delta:+.4f})")
                
                if combined_delta < -0.01:
                    logger.info(f"  Impact: SIGNIFICANT NEGATIVE (component is important)")
                elif combined_delta < 0:
                    logger.info(f"  Impact: Minor negative")
                elif combined_delta > 0.01:
                    logger.info(f"  Impact: POSITIVE (component may be harmful)")
                else:
                    logger.info(f"  Impact: Negligible")
            else:
                logger.info(f"Results: EM={em:.4f}, nDCG={ndcg:.4f}, Combined={combined:.4f}")
            
            logger.info("="*60 + "\n")
            
            return result
            
        except Exception as e:
            logger.error(f"Error ablating {component_name}: {e}")
            raise
    
    def modify_component(self, component_name: str,
                        modification: Dict[str, Any],
                        evaluation_fn: Callable,
                        description: str = "") -> AblationResult:
        """
        Modify a component parameter and evaluate impact.
        
        Args:
            component_name: Name of component to modify
            modification: Dictionary of parameter modifications
            evaluation_fn: Function to evaluate with modifications
            description: Description of the modification
            
        Returns:
            Ablation result
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"MODIFYING: {component_name}")
        logger.info(f"Modifications: {modification}")
        logger.info(f"Description: {description}")
        logger.info(f"{'='*60}")
        
        try:
            em, ndcg, combined = evaluation_fn(self.validation_data)
            
            result = AblationResult(
                experiment_name=f"modify_{component_name}",
                description=description or f"Modified {component_name}: {modification}",
                em_score=em,
                ndcg_score=ndcg,
                combined_score=combined,
                components_modified={component_name: modification}
            )
            
            self.results.append(result)
            
            # Calculate impact
            if self.baseline_result:
                em_delta = em - self.baseline_result.em_score
                ndcg_delta = ndcg - self.baseline_result.ndcg_score
                combined_delta = combined - self.baseline_result.combined_score
                
                logger.info(f"Results:")
                logger.info(f"  EM: {em:.4f} (Δ {em_delta:+.4f})")
                logger.info(f"  nDCG: {ndcg:.4f} (Δ {ndcg_delta:+.4f})")
                logger.info(f"  Combined: {combined:.4f} (Δ {combined_delta:+.4f})")
            else:
                logger.info(f"Results: EM={em:.4f}, nDCG={ndcg:.4f}, Combined={combined:.4f}")
            
            logger.info("="*60 + "\n")
            
            return result
            
        except Exception as e:
            logger.error(f"Error modifying {component_name}: {e}")
            raise
    
    def compare_variants(self, variant_configs: Dict[str, Dict[str, Any]],
                        evaluation_fn: Callable) -> List[AblationResult]:
        """
        Compare multiple system variants.
        
        Args:
            variant_configs: Dictionary of variant name to configuration
            evaluation_fn: Function that takes config and returns scores
            
        Returns:
            List of ablation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARING {len(variant_configs)} VARIANTS")
        logger.info(f"{'='*60}")
        
        variant_results = []
        
        for variant_name, config in variant_configs.items():
            logger.info(f"\nEvaluating variant: {variant_name}")
            logger.info(f"Configuration: {config}")
            
            try:
                em, ndcg, combined = evaluation_fn(config, self.validation_data)
                
                result = AblationResult(
                    experiment_name=f"variant_{variant_name}",
                    description=f"Variant: {variant_name}",
                    em_score=em,
                    ndcg_score=ndcg,
                    combined_score=combined,
                    metadata={'variant_config': config}
                )
                
                variant_results.append(result)
                self.results.append(result)
                
                logger.info(f"Results: EM={em:.4f}, nDCG={ndcg:.4f}, Combined={combined:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating variant {variant_name}: {e}")
                continue
        
        # Rank variants
        ranked = sorted(variant_results, key=lambda x: x.combined_score, reverse=True)
        
        logger.info(f"\n{'='*60}")
        logger.info("VARIANT RANKING")
        logger.info(f"{'='*60}")
        for i, result in enumerate(ranked, 1):
            logger.info(f"{i}. {result.experiment_name}: {result.combined_score:.4f}")
        logger.info(f"{'='*60}\n")
        
        return variant_results
    
    def analyze_component_importance(self) -> Dict[str, float]:
        """
        Analyze importance of each component based on ablation results.
        
        Returns:
            Dictionary of component name to importance score
        """
        if not self.baseline_result:
            logger.warning("No baseline result available")
            return {}
        
        importance_scores = {}
        
        for result in self.results:
            if result.components_removed:
                for component in result.components_removed:
                    # Importance = baseline_score - ablated_score
                    # Positive means component is helpful
                    importance = self.baseline_result.combined_score - result.combined_score
                    importance_scores[component] = importance
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True))
        
        logger.info(f"\n{'='*60}")
        logger.info("COMPONENT IMPORTANCE ANALYSIS")
        logger.info(f"{'='*60}")
        for component, importance in sorted_importance.items():
            impact = "CRITICAL" if importance > 0.05 else "IMPORTANT" if importance > 0.02 else "MODERATE" if importance > 0.01 else "MINOR"
            logger.info(f"{component}: {importance:+.4f} ({impact})")
        logger.info(f"{'='*60}\n")
        
        return sorted_importance
    
    def generate_report(self) -> str:
        """
        Generate comprehensive ablation study report.
        
        Returns:
            Path to report file
        """
        report_file = self.output_dir / "ablation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("ABLATION STUDY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Baseline
            if self.baseline_result:
                f.write("BASELINE SYSTEM\n")
                f.write("-" * 80 + "\n")
                f.write(f"EM Score: {self.baseline_result.em_score:.4f}\n")
                f.write(f"nDCG Score: {self.baseline_result.ndcg_score:.4f}\n")
                f.write(f"Combined Score: {self.baseline_result.combined_score:.4f}\n\n")
            
            # All experiments
            f.write("ABLATION EXPERIMENTS\n")
            f.write("-" * 80 + "\n\n")
            
            for result in self.results:
                if result.experiment_name == "baseline":
                    continue
                
                f.write(f"Experiment: {result.experiment_name}\n")
                f.write(f"Description: {result.description}\n")
                
                if result.components_removed:
                    f.write(f"Removed: {', '.join(result.components_removed)}\n")
                if result.components_modified:
                    f.write(f"Modified: {result.components_modified}\n")
                
                f.write(f"Results:\n")
                f.write(f"  EM: {result.em_score:.4f}\n")
                f.write(f"  nDCG: {result.ndcg_score:.4f}\n")
                f.write(f"  Combined: {result.combined_score:.4f}\n")
                
                if self.baseline_result:
                    delta = result.combined_score - self.baseline_result.combined_score
                    f.write(f"  Impact: {delta:+.4f}\n")
                
                f.write("\n")
            
            # Component importance
            importance = self.analyze_component_importance()
            if importance:
                f.write("COMPONENT IMPORTANCE RANKING\n")
                f.write("-" * 80 + "\n")
                for i, (component, score) in enumerate(importance.items(), 1):
                    f.write(f"{i}. {component}: {score:+.4f}\n")
                f.write("\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Validation samples: {len(self.validation_data)}\n")
            
            if self.baseline_result:
                best_result = max(self.results, key=lambda x: x.combined_score)
                f.write(f"\nBest configuration: {best_result.experiment_name}\n")
                f.write(f"Best score: {best_result.combined_score:.4f}\n")
                improvement = best_result.combined_score - self.baseline_result.combined_score
                f.write(f"Improvement over baseline: {improvement:+.4f}\n")
        
        logger.info(f"Ablation report saved to {report_file}")
        
        # Also save JSON
        json_file = self.output_dir / "ablation_results.json"
        results_data = []
        for result in self.results:
            results_data.append({
                'experiment_name': result.experiment_name,
                'description': result.description,
                'em_score': result.em_score,
                'ndcg_score': result.ndcg_score,
                'combined_score': result.combined_score,
                'components_removed': result.components_removed,
                'components_modified': result.components_modified,
                'metadata': result.metadata
            })
        
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Ablation results saved to {json_file}")
        
        return str(report_file)
