#!/usr/bin/env python3
"""
Quick test with just 5 neurons to verify non-zero results.
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

def main():
    """Quick test with limited neurons."""
    print("Loading dataset...")
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    dataset = dataset.filter(lambda x: x['is_training'] == True)
    
    print("Loading LLM...")
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    print("Loading crosscoder...")
    crosscoder_name = "86u64trx"
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", crosscoder_name, "../../.wandb_artifacts", DEVICE
    )
    
    print("QUICK TEST: Processing normal amount but with fewer samples")
    
    result = compute_shapley_interactions_sequential(
        dataset=dataset,
        llm=llm,
        crosscoder=crosscoder,
        num_stories=1,  # Just 1 story
        layer=0,
        max_features_per_neuron=5,  # Fewer features  
        num_samples=100,  # Fewer samples
        threshold=1e-3,  # Higher threshold to skip more neurons
        verbose=True
    )
    
    print("\n" + "=" * 50)
    print("QUICK TEST RESULTS")
    print("=" * 50)
    print(f"Result shape: {result.shape}")
    print(f"Max value: {result.abs().max().item():.8f}")
    print(f"Min value: {result.abs().min().item():.8f}")
    print(f"Non-zero count: {(result.abs() > 1e-10).sum().item()}")
    print(f"Total elements: {result.numel()}")
    
    # Check for non-zero interactions
    if result.abs().max() > 1e-10:
        print("✅ SUCCESS: Non-zero interactions detected!")
        top_interactions = torch.topk(result.abs().flatten(), k=5)
        print("Top 5 interaction magnitudes:", [f"{x:.8f}" for x in top_interactions.values.tolist()])
    else:
        print("❌ ISSUE: All interactions are still zero")
    
    return result

if __name__ == "__main__":
    result = main()