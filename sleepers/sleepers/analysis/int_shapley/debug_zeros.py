#!/usr/bin/env python3
"""Debug why we're getting all zeros with the config."""

import sys
import os
import torch
import yaml
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from compare_with_config import load_config, setup_device, load_models_and_data

def debug_thresholds():
    """Debug threshold issues."""
    print("=" * 60)
    print("DEBUGGING ZERO INTERACTIONS ISSUE")
    print("=" * 60)
    
    # Load configuration
    config = load_config("comparison_config.yaml")
    device = setup_device(config)
    
    print(f"Config values:")
    print(f"  threshold: {config['shapley_taylor']['threshold']} (type: {type(config['shapley_taylor']['threshold'])})")
    print(f"  small_threshold: {config['shapley_taylor']['small_threshold']} (type: {type(config['shapley_taylor']['small_threshold'])})")
    print(f"  max_features: {config['shapley_taylor']['max_features_per_neuron']}")
    
    # Load models
    dataset, llm, crosscoder = load_models_and_data(config, device)
    
    # Get activations for one story to inspect
    from sleepers.analysis.analysis_utils import get_activations, get_preacts_nocontract
    
    story_text = dataset[0]['text']
    print(f"\nProcessing story: {len(story_text)} chars")
    
    # Get activations
    feature_activations_SH, activations_SMLD = get_activations(story_text, llm, crosscoder)
    enc_acts_BH = feature_activations_SH
    
    preacts = get_preacts_nocontract(
        enc_acts_BH,
        crosscoder.W_dec_HXD,
        crosscoder.b_dec_XD,
        llm,
        block=0,
        bias=True
    )
    
    print(f"\nPreacts shape: {preacts.shape}")
    
    # Sum over sequence length
    story_totals = preacts.sum(dim=0).cpu()  # [d_mlp, hidden_dim]
    print(f"Story totals shape: {story_totals.shape}")
    
    # Check threshold filtering
    threshold = float(config['shapley_taylor']['threshold'])
    small_threshold = float(config['shapley_taylor']['small_threshold'])
    
    print(f"\nThreshold analysis:")
    print(f"  Threshold: {threshold}")
    print(f"  Small threshold: {small_threshold}")
    
    # Check how many neurons pass threshold
    max_activations = story_totals.abs().max(dim=1)[0]  # Max across features for each neuron
    neurons_above_threshold = (max_activations >= threshold).sum()
    
    print(f"  Max activations range: {max_activations.min():.6f} to {max_activations.max():.6f}")
    print(f"  Neurons above threshold ({threshold}): {neurons_above_threshold} / {story_totals.shape[0]}")
    
    if neurons_above_threshold == 0:
        print("❌ NO NEURONS ABOVE THRESHOLD! This explains the zero interactions.")
        print("   Try lowering the threshold to 1e-6 or 1e-4")
    else:
        print("✅ Some neurons above threshold, issue might be elsewhere")
        
        # Check a specific neuron
        neuron_idx = max_activations.argmax().item()
        neuron_features = story_totals[neuron_idx]
        
        print(f"\n  Examining neuron {neuron_idx} (max activation: {max_activations[neuron_idx]:.6f})")
        
        # Check active features
        active_mask = neuron_features.abs() > small_threshold
        num_active = active_mask.sum()
        print(f"  Active features: {num_active}")
        
        if num_active < 2:
            print("❌ Not enough active features for interactions")
        else:
            print("✅ Sufficient active features")

if __name__ == "__main__":
    debug_thresholds()