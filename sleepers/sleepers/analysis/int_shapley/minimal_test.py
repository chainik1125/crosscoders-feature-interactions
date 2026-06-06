#!/usr/bin/env python3
"""
Minimal test to check if we get positive values.
"""

import sys
import os
import torch
from datasets import load_dataset

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from sleepers.scripts.llms import build_llm_lora
from sleepers.scripts.utils import load_crosscoder_from_wandb
from shapley_interactions import compute_shapley_interactions_sequential

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def minimal_test():
    """Super minimal test - just check if we get non-zero results."""
    
    print("MINIMAL TEST: Looking for positive interactions...")
    
    # Load models
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
    
    # Extremely minimal parameters
    result = compute_shapley_interactions_sequential(
        dataset=dataset,
        llm=llm,
        crosscoder=crosscoder,
        num_stories=1,                    # Just 1 story
        layer=0,
        max_features_per_neuron=3,        # Top 3 features only
        num_samples=50,                   # Minimal samples for speed
        threshold=1e-6,                   # Lower threshold
        small_threshold=1e-8,             # Even lower
        device=DEVICE,
        verbose=False,                    # No verbose to reduce output
        value_function_type="gelu",
        max_tokens_per_story=2            # Just 2 tokens!
    )
    
    # Check for any positive values
    positive_count = (result > 1e-10).sum().item()
    negative_count = (result < -1e-10).sum().item() 
    max_positive = result.max().item()
    min_negative = result.min().item()
    
    print(f"Results:")
    print(f"  Positive interactions: {positive_count}")
    print(f"  Negative interactions: {negative_count}")
    print(f"  Max positive value: {max_positive:.8f}")
    print(f"  Min negative value: {min_negative:.8f}")
    print(f"  Total non-zero: {(result.abs() > 1e-10).sum().item()}")
    
    if positive_count > 0:
        print("✅ Found positive interactions!")
    else:
        print("❌ No positive interactions found")
    
    return result

if __name__ == "__main__":
    result = minimal_test()