#!/usr/bin/env python3
"""
Debug nshap integration with a single neuron.
"""

import sys
import os
import torch
import numpy as np
from datasets import load_dataset

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from sleepers.scripts.llms import build_llm_lora
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.analysis.analysis_utils import get_activations, get_preacts_nocontract

import nshap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_nshap_single_neuron():
    """Debug nshap with extracted data from a single neuron."""
    
    print("Loading models...")
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    dataset = dataset.filter(lambda x: x['is_training'] == True)
    
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", "86u64trx", "../../.wandb_artifacts", DEVICE
    )
    
    print("Extracting data for one story...")
    story_text = dataset[0]['text']
    
    # Get activations and preactivations
    feature_activations_SH, activations_SMLD = get_activations(story_text, llm, crosscoder)
    enc_acts_BH = feature_activations_SH  # [seq_len, hidden_dim]
    
    preacts = get_preacts_nocontract(
        enc_acts_BH,              # [seq_len, hidden_dim] 
        crosscoder.W_dec_HXD,     # [hidden, contexts, layers, d_model]
        crosscoder.b_dec_XD,      # [contexts, layers, d_model]
        llm,                      # LLM object
        block=0,                  # Layer 0
        bias=True                 # Include bias terms
    )
    
    print(f"Preacts shape: {preacts.shape}")  # Should be [seq_len, d_mlp, hidden_dim]
    
    # Focus on one neuron with highest activation
    neuron_totals = preacts.sum(dim=0)  # [d_mlp, hidden_dim]
    max_neuron_idx = neuron_totals.abs().sum(dim=1).argmax().item()
    
    neuron_features = neuron_totals[max_neuron_idx]  # [hidden_dim=1536]
    print(f"Selected neuron {max_neuron_idx}, max activation: {neuron_features.abs().max():.6f}")
    
    # Get top features by ABSOLUTE VALUE (both positive and negative)
    active_mask = neuron_features.abs() > 1e-6
    active_indices = torch.nonzero(active_mask).squeeze(-1)
    print(f"Active features: {len(active_indices)}")
    
    # Limit to top 5 by absolute value for debugging
    if len(active_indices) > 5:
        top_k = 5
        top_indices = neuron_features.abs().topk(top_k).indices
        active_indices = active_indices[torch.isin(active_indices, top_indices)]
    
    active_features = neuron_features[active_indices].detach().cpu().numpy()
    print(f"Debug: Using {len(active_indices)} features")
    print(f"Feature values: {active_features}")
    
    # Create simple value function for nshap
    def value_function(x, coalition_indices):
        """Value function for nshap."""
        print(f"Debug: Called with coalition_indices: {coalition_indices}")
        
        # Sum the feature values in the coalition, then apply ReLU (POSTACTIVATION)
        if len(coalition_indices) == 0:
            coalition_sum = 0.0
        else:
            coalition_sum = sum(active_features[i] for i in coalition_indices)
        
        # Apply ReLU to the sum (postactivation)
        output = max(0.0, coalition_sum)
        
        print(f"Debug: Coalition {coalition_indices} -> preactivation {coalition_sum:.6f} -> postactivation {output:.6f}")
        return output
    
    # Test value function manually
    print("\nTesting value function manually:")
    print(f"Empty coalition: {value_function(None, [])}")
    print(f"Single feature [0]: {value_function(None, [0])}")
    print(f"All features: {value_function(None, list(range(len(active_features))))}")
    
    # Create dummy x for nshap
    dummy_x = np.ones(len(active_features))
    print(f"Dummy x shape: {dummy_x.shape}")
    
    # Try nshap computation
    print(f"\nRunning nshap.shapley_taylor with n={min(len(active_features), 3)}...")
    try:
        result = nshap.shapley_taylor(
            dummy_x,
            value_function,
            n=min(len(active_features), 3)  # Limit order for speed
        )
        
        print("✅ nshap succeeded!")
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
        
        # Check if it has values
        if hasattr(result, 'values'):
            values = result.values
            print(f"Values type: {type(values)}")
            if hasattr(values, 'keys'):
                print(f"Keys: {list(values.keys())}")
                for key, val in values.items():
                    print(f"  {key}: {val}")
            else:
                print(f"Values content: {values}")
                
        return result
        
    except Exception as e:
        print(f"❌ nshap failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = debug_nshap_single_neuron()