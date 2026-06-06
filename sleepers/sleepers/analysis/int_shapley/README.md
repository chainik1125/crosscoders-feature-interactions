# Shapley-Taylor Interaction Analysis

This module implements memory-efficient computation of pairwise feature interactions using Shapley-Taylor interaction indices for crosscoder features.

## Overview

The implementation processes stories and neurons sequentially to avoid memory overflow, making it feasible to compute interactions for large-scale crosscoder analysis.

## Key Features

- **Memory efficient**: Processes stories sequentially, peak RAM ~50MB per neuron
- **Scalable**: Handles 3072 neurons × 1536 features without memory issues  
- **Theoretically grounded**: Uses proper Shapley-Taylor interaction indices
- **Configurable**: Adjustable feature limits and sampling parameters

## Files

- `shapley_interactions.py`: Main implementation
- `test_implementation.py`: Test script for compilation and shape verification
- `README.md`: This documentation

## Usage

### Basic Test (2 stories)

```bash
cd /path/to/sleepers/sleepers/analysis/int_shapley
python test_implementation.py
```

### Advanced Usage

```python
from shapley_interactions import compute_shapley_interactions_sequential

# Compute interactions for 100 stories
interactions = compute_shapley_interactions_sequential(
    dataset=dataset,
    llm=llm,
    crosscoder=crosscoder,
    num_stories=100,
    layer=0,
    max_features_per_neuron=15,
    num_samples=800
)

# Result: [1536, 1536] interaction matrix
```

## Expected Shapes

1. **Input preactivations**: `[seq_len, d_mlp, hidden_dim]` per story
2. **Accumulated features**: `[d_mlp, hidden_dim]` across all stories  
3. **Per-neuron interactions**: `[1536, 1536]` (sparse, mostly zeros)
4. **Final result**: `[1536, 1536]` averaged across neurons

## Memory Profile

- **Per story**: ~380MB (128×3072×1536 float32)
- **Accumulated**: ~18MB (3072×1536 float32) 
- **Per neuron computation**: ~9MB (1536×1536 float32)
- **Peak usage**: ~50MB (during Shapley-Taylor computation)

## Dependencies

- `torch`
- `numpy` 
- `tqdm`
- `datasets`
- Existing sleepers codebase (shapley.py, analysis_utils.py, etc.)

## Theory

Uses Shapley-Taylor interaction indices to measure how feature pairs work together beyond their individual contributions. For features i,j:

`interaction[i,j]` = synergistic effect of features i and j on neuron activation

The method aggregates these interactions across neurons to get global feature pair relationships.