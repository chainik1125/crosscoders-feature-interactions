#!/usr/bin/env python3
"""
Profile where computation time is being spent.
"""

import sys
import os
import torch
import time
from datasets import load_dataset

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from sleepers.scripts.llms import build_llm_lora
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.analysis.analysis_utils import get_activations, get_preacts_nocontract

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def time_section(name, start_time):
    """Helper to time sections."""
    elapsed = time.time() - start_time
    print(f"{name}: {elapsed:.3f}s")
    return time.time()

def profile_computation():
    """Profile each step of the computation."""
    
    print("=" * 60)
    print("PROFILING COMPUTATION TIME")
    print("=" * 60)
    
    overall_start = time.time()
    
    # 1. Model loading
    start = time.time()
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    dataset = dataset.filter(lambda x: x['is_training'] == True)
    start = time_section("Dataset loading", start)
    
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    start = time_section("LLM loading", start)
    
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", "86u64trx", "../../.wandb_artifacts", DEVICE
    )
    start = time_section("Crosscoder loading", start)
    
    # 2. Single story processing
    story_text = dataset[0]['text']
    
    feature_activations_SH, _ = get_activations(story_text, llm, crosscoder)
    start = time_section("Feature activations", start)
    
    preacts = get_preacts_nocontract(
        feature_activations_SH,
        crosscoder.W_dec_HXD,
        crosscoder.b_dec_XD,
        llm,
        block=0,
        bias=True
    )
    start = time_section("Preactivations computation", start)
    
    # 3. Process just ONE token-neuron pair with timing
    print(f"\n--- Processing single token-neuron pair ---")
    
    token_idx = 0
    token_preacts = preacts[token_idx]  # [d_mlp, hidden_dim]
    
    # Find a neuron with high activation
    neuron_max_activations = token_preacts.abs().max(dim=1)[0]
    top_neuron_idx = neuron_max_activations.argmax().item()
    neuron_features = token_preacts[top_neuron_idx]
    
    print(f"Selected neuron {top_neuron_idx} with max activation: {neuron_max_activations[top_neuron_idx]:.6f}")
    
    # Time the nshap computation specifically
    try:
        import nshap
        
        # Get top 3 features
        top_3_indices = neuron_features.abs().topk(3).indices
        active_features = neuron_features[top_3_indices].detach().cpu().numpy()
        
        print(f"Top 3 feature values: {active_features}")
        
        # Get MLP weights
        mlp_bias = llm.blocks[0].mlp.b_in
        
        # Create value function
        def value_function(x, coalition_indices):
            if len(coalition_indices) == 0:
                coalition_sum = 0.0
            else:
                coalition_sum = sum(active_features[i] for i in coalition_indices)
            
            preactivation_with_bias = coalition_sum + float(mlp_bias[top_neuron_idx].cpu().detach())
            output = torch.nn.functional.gelu(torch.tensor(preactivation_with_bias))
            return float(output)
        
        start = time.time()
        
        # Test value function calls
        print("Testing value function...")
        for i in range(10):  # Test 10 calls
            result = value_function(None, [0, 1])
        start = time_section("10 value function calls", start)
        
        # Time the actual nshap computation
        print("Running nshap.shapley_taylor...")
        dummy_x = __import__('numpy').ones(len(active_features))
        
        start = time.time()
        shapley_result = nshap.shapley_taylor(
            dummy_x,
            value_function,
            n=min(len(active_features), 3)  # Limit to 3
        )
        start = time_section("nshap.shapley_taylor computation", start)
        
        print(f"nshap result keys: {list(shapley_result.keys()) if hasattr(shapley_result, 'keys') else 'No keys method'}")
        
        # Time the matrix extraction
        start = time.time()
        interaction_matrix = torch.zeros(1536, 1536)
        for i in range(len(active_features)):
            for j in range(len(active_features)):
                if i != j:
                    interaction_key = tuple(sorted([i, j]))
                    if interaction_key in shapley_result:
                        feat_i_idx = top_3_indices[i]
                        feat_j_idx = top_3_indices[j]
                        interaction_matrix[feat_i_idx, feat_j_idx] = shapley_result[interaction_key]
        start = time_section("Interaction matrix extraction", start)
        
        print(f"Non-zero interactions found: {(interaction_matrix.abs() > 1e-10).sum()}")
        print(f"Max interaction: {interaction_matrix.abs().max():.8f}")
        
    except Exception as e:
        print(f"Error in nshap computation: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - overall_start
    print(f"\n--- TOTAL TIME: {total_time:.3f}s ---")
    
    # Estimate full computation time
    estimated_per_neuron = 0.5  # Estimate from single computation
    total_neurons_2_stories = 2 * 128 * 3072
    estimated_total_hours = (total_neurons_2_stories * estimated_per_neuron) / 3600
    print(f"Estimated time for 2 stories × 128 tokens: {estimated_total_hours:.1f} hours")

if __name__ == "__main__":
    profile_computation()