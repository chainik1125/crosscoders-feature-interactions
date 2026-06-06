#!/usr/bin/env python3
"""
Quick test with minimal parameters to verify the fix works.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datasets import load_dataset

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from sleepers.scripts.llms import build_llm_lora
from sleepers.scripts.utils import load_crosscoder_from_wandb
from shapley_interactions import compute_shapley_interactions_sequential

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quick_test(max_neurons=5):
    """Quick test with very small parameters."""
    
    print("=" * 50)
    print(f"QUICK TEST: 1 STORY × 2 TOKENS × TOP 3 FEATURES × FIRST {max_neurons} NEURONS")
    print("=" * 50)
    
    # Load just enough for testing
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
    
    print("Models loaded, running quick test...")
    
    # Very minimal test
    result = compute_shapley_interactions_sequential(
        dataset=dataset,
        llm=llm,
        crosscoder=crosscoder,
        num_stories=1,                    # Just 1 story
        layer=0,
        max_features_per_neuron=5,        # Top 3 features only
        num_samples=100,                  # Minimal samples
        threshold=1e-4,                   # Higher threshold to skip inactive neurons
        small_threshold=1e-6,
        device=DEVICE,
        verbose=True,
        value_function_type="gelu",
        max_tokens_per_story=2,           # Only 2 tokens
        max_neurons_per_token=max_neurons # Lximit neurons
    )
    
    print(f"\n✅ SUCCESS!")
    print(f"Result shape: {result.shape}")
    print(f"Max interaction: {result.abs().max():.8f}")
    print(f"Non-zero interactions: {(result.abs() > 1e-10).sum()}")
    
    # Create scatterplot of non-zero interactions
    create_interaction_scatterplot(result, max_neurons)
    
    return result

def create_interaction_scatterplot(interactions, max_neurons):
    """Create and save a scatterplot of non-zero feature interactions."""
    
    print(f"\nCreating scatterplot...")
    
    # Find non-zero interactions
    nonzero_mask = interactions.abs() > 1e-10
    nonzero_indices = torch.nonzero(nonzero_mask)
    nonzero_values = interactions[nonzero_mask]
    
    if len(nonzero_values) == 0:
        print("No non-zero interactions to plot")
        return
    
    # Extract feature i and j indices
    feature_i = nonzero_indices[:, 0].cpu().numpy()
    feature_j = nonzero_indices[:, 1].cpu().numpy()
    values = nonzero_values.cpu().numpy()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Separate positive and negative interactions
    positive_mask = values > 0
    negative_mask = values < 0
    
    if positive_mask.any():
        plt.scatter(feature_i[positive_mask], feature_j[positive_mask], 
                   c=values[positive_mask], cmap='Reds', alpha=0.7, 
                   label=f'Positive ({positive_mask.sum()})', s=50)
    
    if negative_mask.any():
        plt.scatter(feature_i[negative_mask], feature_j[negative_mask], 
                   c=values[negative_mask], cmap='Blues_r', alpha=0.7, 
                   label=f'Negative ({negative_mask.sum()})', s=50)
    
    plt.colorbar(label='Interaction Strength')
    plt.xlabel('Feature i Index')
    plt.ylabel('Feature j Index')
    plt.title(f'Shapley-Taylor Feature Interactions\n'
              f'{max_neurons} neurons, {len(nonzero_values)} non-zero interactions\n'
              f'Range: [{values.min():.6f}, {values.max():.6f}]')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line (no self-interactions expected)
    max_idx = max(feature_i.max(), feature_j.max())
    plt.plot([0, max_idx], [0, max_idx], 'k--', alpha=0.3, label='Diagonal')
    
    # Save to large_files/graphs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"shapley_interactions_{max_neurons}neurons_{timestamp}.png"
    save_path = os.path.join("..", "..", "large_files", "graphs", filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Scatterplot saved to: {save_path}")
    print(f"  Positive interactions: {positive_mask.sum()}")
    print(f"  Negative interactions: {negative_mask.sum()}")
    print(f"  Value range: [{values.min():.6f}, {values.max():.6f}]")

if __name__ == "__main__":
    import sys
    max_neurons = 100  # Default to first 5 neurons
    if len(sys.argv) > 1:
        max_neurons = int(sys.argv[1])
    
    result = quick_test(max_neurons)