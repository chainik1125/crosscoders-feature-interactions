#!/usr/bin/env python3
"""
Test the token-by-token Shapley implementation with 2 stories (128x2 tokens).
Uses top 5 features per neuron as requested.
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
from shapley_interactions import compute_shapley_interactions_sequential, compare_value_functions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_two_stories():
    """Test the implementation with exactly 2 stories and top 5 features."""
    
    print("=" * 80)
    print("TESTING SHAPLEY INTERACTIONS: 2 STORIES × 128 TOKENS × TOP 5 FEATURES")
    print("=" * 80)
    
    print("\n1. Loading models and data...")
    
    # Load dataset
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    dataset = dataset.filter(lambda x: x['is_training'] == True)
    
    print(f"Dataset loaded, total examples: {len(dataset)}")
    
    # Load LLM
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    print(f"LLM loaded on device: {llm.cfg.device}")
    print(f"Model config - d_mlp: {llm.cfg.d_mlp}, d_model: {llm.cfg.d_model}")
    
    # Load crosscoder
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", "86u64trx", "../../.wandb_artifacts", DEVICE
    )
    
    print("Crosscoder loaded")
    print(f"Crosscoder shapes - W_dec: {crosscoder.W_dec_HXD.shape}, b_dec: {crosscoder.b_dec_XD.shape}")
    
    print("\n2. Testing with GELU value function...")
    
    # Test with GELU approach
    gelu_interactions = compute_shapley_interactions_sequential(
        dataset=dataset,
        llm=llm,
        crosscoder=crosscoder,
        num_stories=2,                    # Exactly 2 stories
        layer=0,                          # Test layer 0
        max_features_per_neuron=5,        # Top 5 features as requested
        num_samples=500,                  # Reasonable sample size
        threshold=1e-6,                   # Standard threshold
        small_threshold=1e-8,             # Small threshold
        device=DEVICE,
        verbose=True,
        value_function_type="gelu",       # GELU value function
        max_tokens_per_story=128          # Full 128 tokens per story
    )
    
    print("\n3. Analyzing results...")
    
    # Analyze the results
    print(f"Final interaction matrix shape: {gelu_interactions.shape}")
    print(f"Expected shape (1536, 1536): {gelu_interactions.shape == (1536, 1536)}")
    
    # Statistics
    max_interaction = gelu_interactions.abs().max().item()
    mean_interaction = gelu_interactions.abs().mean().item()
    nonzero_count = (gelu_interactions.abs() > 1e-8).sum().item()
    total_entries = gelu_interactions.numel()
    sparsity = 1.0 - (nonzero_count / total_entries)
    
    print(f"\nInteraction Statistics:")
    print(f"  Max absolute interaction: {max_interaction:.8f}")
    print(f"  Mean absolute interaction: {mean_interaction:.8f}")
    print(f"  Non-zero interactions: {nonzero_count:,}")
    print(f"  Total entries: {total_entries:,}")
    print(f"  Sparsity: {sparsity:.4f} ({100*sparsity:.2f}%)")
    
    # Check for all-negative issue
    positive_count = (gelu_interactions > 1e-8).sum().item()
    negative_count = (gelu_interactions < -1e-8).sum().item()
    
    print(f"  Positive interactions: {positive_count:,}")
    print(f"  Negative interactions: {negative_count:,}")
    
    if negative_count > 0 and positive_count == 0:
        print("  ⚠️  WARNING: All interactions are negative!")
    elif positive_count > 0 and negative_count > 0:
        print("  ✅ Good: Found both positive and negative interactions")
    elif positive_count > 0:
        print("  ✅ Good: Found positive interactions")
    else:
        print("  ❌ Problem: No significant interactions found")
    
    # Show top interactions
    print(f"\nTop 10 interactions by absolute value:")
    flat_interactions = gelu_interactions.flatten()
    abs_interactions = flat_interactions.abs()
    top_indices = abs_interactions.topk(10).indices
    
    for i, idx in enumerate(top_indices):
        row = idx // 1536
        col = idx % 1536
        value = flat_interactions[idx].item()
        print(f"  {i+1:2d}. Feature {row:4d} ↔ Feature {col:4d}: {value:10.8f}")
    
    print(f"\n4. Estimating computational cost...")
    
    # Estimate how many token-neuron pairs were processed
    # This would be num_stories × tokens_per_story × active_neurons_per_token
    estimated_pairs = 2 * 128  # stories × tokens
    print(f"  Total tokens processed: {estimated_pairs}")
    print(f"  Neurons per token: {llm.cfg.d_mlp}")
    print(f"  Max potential token-neuron pairs: {estimated_pairs * llm.cfg.d_mlp:,}")
    print(f"  Features per active neuron: 5")
    
    return gelu_interactions

def test_comparison():
    """Test comparison between GELU and MLP output approaches."""
    
    print("\n" + "=" * 80)
    print("BONUS: COMPARING GELU VS MLP OUTPUT VALUE FUNCTIONS")
    print("=" * 80)
    
    # Load models (reuse from previous test or reload)
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
    
    # Compare both approaches with smaller parameters for speed
    comparison_results = compare_value_functions(
        dataset=dataset,
        llm=llm,
        crosscoder=crosscoder,
        num_stories=1,                    # Just 1 story for comparison
        layer=0,
        max_features_per_neuron=5,        # Top 5 features
        num_samples=200,                  # Smaller for speed
        max_tokens_per_story=20,          # Just 20 tokens for speed
        verbose=True
    )
    
    return comparison_results

if __name__ == "__main__":
    try:
        print("Starting test...")
        
        # Main test with 2 stories
        gelu_result = test_two_stories()
        
        # Optional: comparison test
        print(f"\nWould you like to run the comparison test? (smaller scale)")
        print("This will compare GELU vs MLP output approaches...")
        
        # Uncomment the next line if you want to run the comparison
        # comparison_result = test_comparison()
        
        print(f"\n{'='*80}")
        print("TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()