#!/usr/bin/env python3
"""Test with exact parameters from the working quick_compare.py."""

import sys
import os
import torch
import yaml

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from compare_with_config import load_config, setup_device, load_models_and_data, compute_shapley_interactions

def test_working_parameters():
    """Test with exact parameters that worked in quick_compare.py."""
    print("=" * 60)
    print("TESTING WITH WORKING PARAMETERS")
    print("=" * 60)
    
    # Load base config but override with working parameters
    config = load_config("comparison_config.yaml")
    
    # Override with exact working parameters
    config['comparison']['num_stories'] = 1
    config['comparison']['layer'] = 0
    config['shapley_taylor']['max_features_per_neuron'] = 5
    config['shapley_taylor']['num_samples'] = 100
    config['shapley_taylor']['threshold'] = 1e-2  # This worked!
    config['shapley_taylor']['small_threshold'] = 1e-8
    config['shapley_taylor']['verbose'] = False
    
    print("Using exact working parameters:")
    print(f"  num_stories: {config['comparison']['num_stories']}")
    print(f"  max_features_per_neuron: {config['shapley_taylor']['max_features_per_neuron']}")
    print(f"  threshold: {config['shapley_taylor']['threshold']}")
    print(f"  threshold type: {type(config['shapley_taylor']['threshold'])}")
    
    device = setup_device(config)
    dataset, llm, crosscoder = load_models_and_data(config, device)
    
    # Test the Shapley computation
    try:
        result = compute_shapley_interactions(dataset, llm, crosscoder, config)
        
        print(f"\n✅ SUCCESS!")
        print(f"Result shape: {result.shape}")
        print(f"Max value: {result.abs().max():.6f}")
        print(f"Non-zero count: {(result.abs() > 1e-8).sum()}")
        
        return result
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_working_parameters()