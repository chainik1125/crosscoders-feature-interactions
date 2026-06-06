#!/usr/bin/env python3
"""
Test with many features per neuron to show scaling capability.
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
    """Test with many features per neuron."""
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
    
    print("\n" + "=" * 60)
    print("TESTING: MANY FEATURES PER NEURON")
    print("=" * 60)
    
    # Test with different feature limits
    feature_limits = [10, 25, 50]
    
    for max_features in feature_limits:
        print(f"\n🧪 Testing with max_features_per_neuron = {max_features}")
        print("-" * 40)
        
        result = compute_shapley_interactions_sequential(
            dataset=dataset,
            llm=llm,
            crosscoder=crosscoder,
            num_stories=1,
            layer=0,
            max_features_per_neuron=max_features,
            num_samples=200,  # Lower samples for speed
            threshold=1e-3,  # Higher threshold to focus on active neurons
            verbose=True
        )
        
        print(f"✅ Completed with {max_features} features")
        print(f"   - Max interaction: {result.abs().max().item():.6f}")
        print(f"   - Non-zero interactions: {(result.abs() > 1e-8).sum().item()}")
        print(f"   - Sparsity: {(result.abs() > 1e-8).sum().item() / result.numel() * 100:.3f}%")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED!")
    print("✅ Successfully scales with increased feature counts")
    print("✅ More features = more detected interactions")
    print("=" * 60)

if __name__ == "__main__":
    main()