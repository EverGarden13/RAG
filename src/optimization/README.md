# Performance Optimization System

This module provides comprehensive performance optimization capabilities for the RAG system, including hyperparameter tuning, ablation studies, and optimization pipelines targeting top-tier ranking performance.

## Components

### 1. Hyperparameter Tuner (`hyperparameter_tuner.py`)

Implements multiple search strategies for finding optimal hyperparameters:

- **Grid Search**: Exhaustive search over parameter space
- **Random Search**: Random sampling for faster exploration
- **Bayesian Optimization**: Intelligent search using past results

**Key Features:**
- Configurable parameter spaces (categorical, continuous, integer)
- Automatic result tracking and ranking
- JSON and text output formats
- Support for both retrieval and generation parameters

**Example Usage:**
```python
from src.optimization.hyperparameter_tuner import HyperparameterTuner, HyperparameterSpace

# Define parameter spaces
param_spaces = {
    'k_retrieve': HyperparameterSpace(
        name='k_retrieve',
        values=[50, 100, 150, 200],
        param_type='categorical'
    ),
    'temperature': HyperparameterSpace(
        name='temperature',
        values=[0.0, 0.1, 0.2, 0.3],
        param_type='categorical'
    )
}

# Create tuner
tuner = HyperparameterTuner(
    evaluation_fn=my_eval_function,
    validation_data=validation_set,
    output_dir="./tuning_results"
)

# Run optimization
best_result = tuner.bayesian_optimization(param_spaces, n_iterations=20)
print(f"Best params: {best_result.params}")
print(f"Best score: {best_result.combined_score:.4f}")
```

### 2. Ablation Study Framework (`ablation_study.py`)

Analyzes the contribution of individual components to system performance:

- **Component Removal**: Test impact of removing features
- **Component Modification**: Test impact of parameter changes
- **Variant Comparison**: Compare multiple system configurations
- **Importance Analysis**: Rank components by contribution

**Key Features:**
- Automatic baseline comparison
- Component importance scoring
- Comprehensive reporting (JSON + text)
- Support for complex ablation experiments

**Example Usage:**
```python
from src.optimization.ablation_study import AblationStudyFramework

# Create framework
framework = AblationStudyFramework(
    baseline_fn=baseline_evaluation,
    validation_data=validation_set,
    output_dir="./ablation_results"
)

# Run baseline
baseline = framework.run_baseline()

# Ablate a component
framework.ablate_component(
    component_name="bm25_retrieval",
    evaluation_fn=eval_without_bm25,
    description="System without BM25 sparse retrieval"
)

# Analyze importance
importance = framework.analyze_component_importance()

# Generate report
report_path = framework.generate_report()
```

### 3. Optimization Pipeline (`optimization_pipeline.py`)

Orchestrates complete optimization workflow:

1. **Baseline Evaluation**: Establish starting performance
2. **Retrieval Optimization**: Tune retrieval parameters
3. **Generation Optimization**: Tune generation parameters
4. **Joint Optimization**: Optimize all parameters together
5. **Ablation Study**: Analyze component contributions
6. **Final Evaluation**: Validate best configuration

**Key Features:**
- Multi-phase optimization strategy
- Automatic result tracking and comparison
- Configurable optimization modes (quick/comprehensive)
- Detailed reporting and visualization

**Example Usage:**
```python
from src.optimization.optimization_pipeline import (
    OptimizationPipeline,
    create_comprehensive_optimization_config
)

# Create configuration
config = create_comprehensive_optimization_config()

# Create pipeline
pipeline = OptimizationPipeline(
    rag_system_factory=my_rag_factory,
    validation_data=validation_set,
    config=config
)

# Run full optimization
results = pipeline.run_full_optimization()

print(f"Baseline: {results['baseline']['combined']:.4f}")
print(f"Optimized: {results['final_results']['combined']:.4f}")
print(f"Best params: {results['best_params']}")
```

## Tunable Parameters

### Retrieval Parameters

- `k_retrieve`: Number of documents to retrieve initially (50-200)
- `k_final`: Final number of documents to return (fixed at 10)
- `fusion_method`: Method for combining retrievers (rrf, combsum, weighted)
- `bm25_weight`: Weight for BM25 retriever (0.0-1.0)
- `dense_weight`: Weight for dense retriever (0.0-1.0)

### Generation Parameters

- `temperature`: Sampling temperature (0.0-0.5)
- `max_tokens`: Maximum tokens to generate (256-768)
- `top_p`: Nucleus sampling parameter (0.8-1.0)
- `max_context_length`: Maximum context length (1500-3000)

## Running Optimization

### Quick Optimization (Testing)

For rapid testing with a subset of data:

```bash
python run_optimization.py --mode quick --sample-size 100 --n-iterations 10
```

This will:
- Use 100 validation samples
- Run 10 tuning iterations
- Skip ablation study
- Complete in ~15-30 minutes

### Comprehensive Optimization (Production)

For full optimization targeting top-tier performance:

```bash
python run_optimization.py --mode comprehensive
```

This will:
- Use all validation data
- Run 30 tuning iterations per phase
- Include full ablation study
- Complete in ~2-4 hours

### Custom Optimization

```bash
python run_optimization.py \
    --mode comprehensive \
    --sample-size 500 \
    --n-iterations 25 \
    --output-dir ./my_optimization
```

## Output Files

The optimization system generates several output files:

```
optimization_results/
├── optimization_results.json       # Complete results in JSON
├── optimization_summary.txt        # Human-readable summary
├── retrieval_tuning/
│   ├── bayesian_optimization_results.json
│   └── bayesian_optimization_summary.txt
├── generation_tuning/
│   ├── bayesian_optimization_results.json
│   └── bayesian_optimization_summary.txt
├── joint_tuning/
│   ├── bayesian_optimization_results.json
│   └── bayesian_optimization_summary.txt
└── ablation_study/
    ├── ablation_results.json
    └── ablation_report.txt
```

## Optimization Strategies

### 1. Sequential Optimization

Optimize components in sequence:
1. Retrieval parameters first
2. Generation parameters second
3. Joint refinement third

**Advantages:**
- Faster convergence
- Easier to debug
- Clear component contributions

### 2. Joint Optimization

Optimize all parameters simultaneously:

**Advantages:**
- Finds global optimum
- Accounts for parameter interactions
- Better final performance

**Disadvantages:**
- Slower convergence
- Requires more iterations
- Harder to interpret

### 3. Ablation-Guided Optimization

Use ablation study results to guide optimization:
1. Identify most important components
2. Focus tuning on high-impact parameters
3. Remove or simplify low-impact components

## Performance Targets

Based on project requirements:

- **Minimum Target**: Top 30% ranking (14/20 points)
  - Combined score: ~0.60-0.65
  
- **Good Target**: Top 20% ranking (16/20 points)
  - Combined score: ~0.65-0.70
  
- **Excellent Target**: Top 10% ranking (18-20/20 points)
  - Combined score: ~0.70+

## Best Practices

1. **Start with Baseline**: Always establish baseline performance first
2. **Use Validation Set**: Never optimize on test data
3. **Track Everything**: Save all intermediate results
4. **Iterate Gradually**: Make incremental improvements
5. **Validate Changes**: Test each optimization phase
6. **Monitor Overfitting**: Watch for validation/test divergence
7. **Document Decisions**: Record why certain parameters were chosen

## Troubleshooting

### Optimization Not Improving

- Check if parameter ranges are appropriate
- Increase number of iterations
- Try different search strategies
- Verify evaluation function is working correctly

### Slow Optimization

- Reduce validation sample size
- Use quick mode for testing
- Parallelize evaluations if possible
- Cache intermediate results

### Inconsistent Results

- Increase validation sample size
- Use multiple random seeds
- Average over multiple runs
- Check for data leakage

## Integration with Main System

The optimized parameters can be integrated into the main system:

```python
# Load optimized parameters
import json
with open('optimization_results/optimization_results.json') as f:
    results = json.load(f)

best_params = results['best_params']

# Update system configuration
from config import CONFIG
CONFIG.retrieval.k_retrieve = best_params['k_retrieve']
CONFIG.generation.temperature = best_params['temperature']
# ... etc
```

## Examples

See `examples/optimization_example.py` for complete working examples of:
- Hyperparameter tuning
- Ablation studies
- Parameter comparison

Run examples:
```bash
python examples/optimization_example.py
```

## Requirements

The optimization system requires:
- Validation dataset (HQ-small validation split)
- Working RAG system
- OpenRouter API key
- Sufficient compute resources

## References

- Hyperparameter Optimization: Bergstra & Bengio (2012)
- Ablation Studies: Melis et al. (2017)
- Bayesian Optimization: Snoek et al. (2012)
- Reciprocal Rank Fusion: Cormack et al. (2009)
