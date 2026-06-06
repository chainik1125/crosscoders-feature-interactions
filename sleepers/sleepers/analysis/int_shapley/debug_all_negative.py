#!/usr/bin/env python3
"""Debug why all nshap values are negative."""

import sys
import os
import torch
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from compare_with_config import load_config, setup_device, load_models_and_data
from sleepers.analysis.analysis_utils import get_activations, get_preacts_nocontract

def debug_negative_values():
    """Debug why all nshap values are negative."""
    print("=" * 60)
    print("DEBUGGING ALL-NEGATIVE NSHAP VALUES")
    print("=" * 60)
    
    # Load config and models
    config = load_config("conservative_test.yaml")
    device = setup_device(config)
    dataset, llm, crosscoder = load_models_and_data(config, device)
    
    # Get data for one story
    story_text = dataset[0]['text']
    feature_activations_SH, _ = get_activations(story_text, llm, crosscoder)
    
    preacts = get_preacts_nocontract(
        feature_activations_SH,
        crosscoder.W_dec_HXD,
        crosscoder.b_dec_XD,
        llm,
        block=0,
        bias=True
    )
    
    # Sum over sequence length
    story_totals = preacts.sum(dim=0).cpu()  # [d_mlp, hidden_dim]
    
    # Find a neuron with high activation
    max_activations = story_totals.abs().max(dim=1)[0]
    neuron_idx = max_activations.argmax().item()
    neuron_features = story_totals[neuron_idx]
    
    print(f"Examining neuron {neuron_idx}")
    print(f"Max activation: {max_activations[neuron_idx]:.6f}")
    
    # Get top features
    active_mask = neuron_features.abs() > 1e-8
    active_indices = torch.nonzero(active_mask).squeeze(-1)
    
    # Limit to top 5 for debugging
    if len(active_indices) > 5:
        top_indices = neuron_features.abs().topk(5).indices
        active_indices = active_indices[torch.isin(active_indices, top_indices)]
    
    active_features = neuron_features[active_indices].detach().cpu().numpy()
    
    print(f"Active features: {len(active_features)}")
    print(f"Feature values: {active_features}")
    print(f"Feature sum (preactivation): {active_features.sum():.6f}")
    
    # Get MLP bias for this neuron
    mlp_bias_tensor = llm.blocks[0].mlp.b_in  # [d_mlp]
    mlp_bias = float(mlp_bias_tensor[neuron_idx].cpu())
    
    print(f"MLP bias for neuron {neuron_idx}: {mlp_bias:.6f}")
    
    # Test the value function manually WITH BIAS
    import nshap
    
    def value_function(x, coalition_indices):
        if len(coalition_indices) == 0:
            coalition_sum = 0.0
        else:
            coalition_sum = sum(active_features[i] for i in coalition_indices)
        
        # Add bias and apply ReLU (postactivation)
        output = max(0.0, coalition_sum + mlp_bias)
        return output
    
    # Test some coalitions manually
    print(f"\nTesting value function WITH BIAS:")
    print(f"Empty coalition: {value_function(None, [])}")
    print(f"Single feature [0]: {value_function(None, [0])}")
    print(f"All features: {value_function(None, list(range(len(active_features))))}")
    print(f"Two features [0,1]: {value_function(None, [0, 1])}")
    
    # Check if the issue is in the ReLU
    print(f"\nPreactivation analysis WITH BIAS:")
    print(f"Sum of all features: {sum(active_features):.6f}")
    print(f"Sum + bias: {sum(active_features) + mlp_bias:.6f}")
    print(f"ReLU of (sum + bias): {max(0.0, sum(active_features) + mlp_bias):.6f}")
    
    # Check if bias fixes the issue
    if sum(active_features) + mlp_bias <= 0:
        print("❌ STILL NEGATIVE: Even with bias, total is non-positive!")
        print("   This means the bias is not large enough to overcome negative features.")
    else:
        print("✅ BIAS FIXES IT: Sum + bias is positive, should get non-zero interactions!")
    
    # Try with different neurons WITH BIAS
    print(f"\nTrying different neurons WITH BIAS:")
    for i in range(10):
        test_neuron_idx = max_activations.topk(10).indices[i].item()
        test_features = story_totals[test_neuron_idx]
        test_sum = test_features.sum().item()
        test_bias = float(mlp_bias_tensor[test_neuron_idx].cpu())
        test_sum_with_bias = test_sum + test_bias
        test_relu = max(0.0, test_sum_with_bias)
        print(f"Neuron {test_neuron_idx}: sum={test_sum:.6f}, bias={test_bias:.6f}, sum+bias={test_sum_with_bias:.6f}, ReLU={test_relu:.6f}")
    
    return neuron_features, active_features

if __name__ == "__main__":
    result = debug_negative_values()