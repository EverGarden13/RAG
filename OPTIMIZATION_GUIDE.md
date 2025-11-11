# RAG System Performance Optimization Guide

This guide explains how to use the performance optimization system to achieve top-tier ranking on the HotpotQA benchmark.

## Overview

The optimization system provides three main capabilities:

1. **Hyperparameter Tuning**: Find optimal parameters for retrieval and generation
2. **Ablation Studies**: Analyze component contributions to system performance
3. **Optimization Pipeline**: Orchestrate complete optimization workflow

## Quick Start

### 1. Run Example (Test the System)

```bash
python examples/optimization_example.py
```

This demonstrates:
- Hyperparameter tuning with grid search
- Ablation study with component removal
- Parameter configuration comparison

### 2. Quick Optimization (15-30 minutes)

```bash
python run_optimization.py --mode quick --sample-size 100
```

This runs a fast optimization using:
- 100 validation samples
- 10 tuning iterations
- Random search strategy
- No ablation study

### 3. Full Optimization (2-4 hours)

```bash
python run_optimization.py --mode comprehensive
```

This runs complete optimization:
- All validation data (~1500 samples)
- 30 tuning iterations per phase
- Bayesian optimization
- Full ablation study

## Optimization Workflow

### Phase 1: Baseline Evaluation

Establishes starting performance with default parameters:
- k_retrieve: 100
- temperature: 0.1
- fusion_method: rrf

**Expected baseline**: Combined score ~0.50-0.60

### Phase 2: Retrieval Optimization

Tunes retrieval parameters:
- `k_retrieve`: Number of documents to retrieve (50-200)
- `fusion_method`: Combination strategy (rrf, combsum, weighted)
- `bm25_weight`: Weight for sparse retrieval (0.2-0.6)
- `dense_weight`: Weight for dense retrieval (0.4-0.8)

**Expected improvement**: +0.05-0.10

### Phase 3: Generation Optimization

Tunes generation parameters:
- `temperature`: Sampling temperature (0.0-0.3)
- `max_tokens`: Maximum output length (256-768)
- `top_p`: Nucleus sampling (0.8-1.0)
- `max_context_length`: Context window (1500-3000)

**Expected improvement**: +0.02-0.05

### Phase 4: Joint Optimization

Optimizes all parameters together to find global optimum.

**Expected improvement**: +0.01-0.03

### Phase 5: Ablation Study

Analyzes component importance:
- Tests impact of removing BM25
- Tests impact of removing dense retrieval
- Tests impact of removing multi-turn
- Tests impact of removing agentic features

**Insights**: Identifies which components are critical vs. optional

### Phase 6: Final Evaluation

Validates best configuration on full validation set.

**Target**: Combined score 0.65-0.75 for top-tier ranking

## Understanding Results

### Output Files

```
optimization_results/
├── optimization_results.json       # Complete results
├── optimization_summary.txt        # Human-readable summary
├── retrieval_tuning/              # Retrieval optimization results
├── generation_tuning/             # Generation optimization results
├── joint_tuning/                  # Joint optimization results
└── ablation_study/                # Ablation study results
```

### Reading the Summary

```
BASELINE RESULTS
--------------------------------------------------------------------------------
EM: 0.5500
nDCG: 0.6200
Combined: 0.5850

FINAL OPTIMIZED RESULTS
--------------------------------------------------------------------------------
EM: 0.6800
nDCG: 0.7100
Combined: 0.6950

IMPROVEMENT: +0.1100 (+18.80%)
```

### Interpreting Scores

- **EM (Exact Match)**: Answer accuracy (0.0-1.0)
- **nDCG@10**: Retrieval quality (0.0-1.0)
- **Combined**: Average of EM and nDCG

**Performance Tiers:**
- 0.50-0.60: Baseline (50-60% ranking)
- 0.60-0.65: Good (top 40%)
- 0.65-0.70: Very Good (top 20%)
- 0.70+: Excellent (top 10%)

## Optimization Strategies

### Strategy 1: Fast Iteration

For rapid testing and development:

```bash
python run_optimization.py \
    --mode quick \
    --sample-size 50 \
    --n-iterations 5
```

**Time**: 5-10 minutes
**Use case**: Testing changes, debugging

### Strategy 2: Balanced Optimization

For good results with reasonable time:

```bash
python run_optimization.py \
    --mode comprehensive \
    --sample-size 500 \
    --n-iterations 20
```

**Time**: 1-2 hours
**Use case**: Development, iterative improvement

### Strategy 3: Maximum Performance

For best possible results:

```bash
python run_optimization.py \
    --mode comprehensive \
    --n-iterations 30
```

**Time**: 3-4 hours
**Use case**: Final optimization, competition submission

## Advanced Usage

### Custom Parameter Spaces

Edit `src/optimization/hyperparameter_tuner.py`:

```python
def create_custom_param_spaces():
    return {
        'k_retrieve': HyperparameterSpace(
            name='k_retrieve',
            values=[75, 100, 125, 150, 175],  # Custom values
            param_type='categorical'
        ),
        # Add more parameters...
    }
```

### Custom Ablation Studies

Create custom ablation experiments:

```python
from src.optimization.ablation_study import AblationStudyFramework

framework = AblationStudyFramework(...)

# Test without specific component
framework.ablate_component(
    component_name="colbert_retrieval",
    evaluation_fn=eval_without_colbert,
    description="System without ColBERT multi-vector retrieval"
)
```

### Integration with Main System

Apply optimized parameters:

```python
import json

# Load optimized parameters
with open('optimization_results/optimization_results.json') as f:
    results = json.load(f)

best_params = results['best_params']

# Update config
from config import CONFIG
CONFIG.retrieval.k_retrieve = best_params['k_retrieve']
CONFIG.retrieval.fusion_method = best_params['fusion_method']
CONFIG.generation.temperature = best_params['temperature']
CONFIG.generation.max_tokens = best_params['max_tokens']
```

## Troubleshooting

### Issue: Optimization Not Improving

**Symptoms**: Final score similar to baseline

**Solutions**:
1. Increase number of iterations
2. Expand parameter search ranges
3. Try different search strategy (bayesian vs. random)
4. Check if evaluation function is working correctly

### Issue: Slow Optimization

**Symptoms**: Takes too long to complete

**Solutions**:
1. Reduce validation sample size
2. Use quick mode for testing
3. Reduce number of iterations
4. Use random search instead of grid search

### Issue: Inconsistent Results

**Symptoms**: Different runs give very different results

**Solutions**:
1. Increase validation sample size
2. Use more iterations
3. Set random seed for reproducibility
4. Average over multiple runs

### Issue: Out of Memory

**Symptoms**: System crashes during optimization

**Solutions**:
1. Reduce batch size in retrieval
2. Reduce validation sample size
3. Use smaller embedding models
4. Clear cache between iterations

## Best Practices

### 1. Start Small

Begin with quick optimization to verify everything works:
```bash
python run_optimization.py --mode quick --sample-size 50
```

### 2. Iterate Gradually

Don't jump to full optimization immediately:
1. Quick optimization (50 samples)
2. Medium optimization (200 samples)
3. Full optimization (all samples)

### 3. Monitor Progress

Check intermediate results:
```bash
# View tuning results
cat optimization_results/retrieval_tuning/bayesian_optimization_summary.txt

# View ablation results
cat optimization_results/ablation_study/ablation_report.txt
```

### 4. Save Everything

Keep all optimization runs:
```bash
python run_optimization.py --output-dir ./opt_run_1
python run_optimization.py --output-dir ./opt_run_2
```

### 5. Validate on Test Set

After optimization, validate on test set:
```bash
python main.py --config optimized_config.json --test
```

## Performance Targets

Based on project requirements:

### Minimum (Top 30%)
- **Target**: 14/20 points
- **Combined Score**: 0.60-0.65
- **Strategy**: Basic optimization, 10-15 iterations

### Good (Top 20%)
- **Target**: 16/20 points
- **Combined Score**: 0.65-0.70
- **Strategy**: Comprehensive optimization, 20-25 iterations

### Excellent (Top 10%)
- **Target**: 18-20/20 points
- **Combined Score**: 0.70+
- **Strategy**: Full optimization + ablation + manual tuning

## Tips for Top Performance

### 1. Retrieval Quality is Critical

Focus on retrieval optimization:
- Test multiple fusion methods
- Tune retrieval weights carefully
- Consider using more retrievers

### 2. Temperature Matters

Lower temperature (0.0-0.1) typically works better for factual QA:
- More deterministic outputs
- Better exact match scores
- Less hallucination

### 3. Context Length Trade-off

Balance context length:
- More context = more information
- Too much context = noise and confusion
- Optimal: 2000-2500 tokens

### 4. Hybrid Retrieval Wins

Combining multiple retrieval methods usually outperforms single methods:
- BM25 + Dense is strong baseline
- RRF fusion typically best
- Tune weights for your data

### 5. Don't Overfit

Watch for overfitting to validation set:
- Use large validation sample
- Monitor validation vs. test gap
- Keep some data held out

## Example Workflow

Complete optimization workflow:

```bash
# 1. Test the system
python examples/optimization_example.py

# 2. Quick optimization to verify
python run_optimization.py --mode quick --sample-size 100

# 3. Medium optimization for development
python run_optimization.py --mode comprehensive --sample-size 500 --n-iterations 20

# 4. Full optimization for final submission
python run_optimization.py --mode comprehensive --n-iterations 30

# 5. Apply best parameters
# Edit config.py with best parameters from optimization_results/

# 6. Generate final predictions
python main.py --test --output test_prediction.jsonl

# 7. Validate results
python src/evaluation/metrics.py --predictions test_prediction.jsonl --references test_data.jsonl
```

## Support

For issues or questions:
1. Check the logs in `./logs/`
2. Review the README in `src/optimization/`
3. Run examples to verify setup
4. Check diagnostics with `getDiagnostics` tool

## References

- Hyperparameter Optimization: Bergstra & Bengio (2012)
- Ablation Studies: Melis et al. (2017)
- Bayesian Optimization: Snoek et al. (2012)
- RAG Systems: Lewis et al. (2020)
