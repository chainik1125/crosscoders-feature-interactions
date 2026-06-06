#!/usr/bin/env python3
"""
Quick comparison between Shapley-Taylor and existing interaction methods.
Optimized for speed with minimal parameters.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from sleepers.scripts.llms import build_llm_lora
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.analysis.analysis_utils import feature_interactions_mlp
from shapley_interactions import compute_shapley_interactions_sequential

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quick_comparison():
    """Quick comparison with minimal parameters for speed."""
    print("=" * 60)
    print("QUICK INTERACTION METHOD COMPARISON")
    print("=" * 60)
    
    # Load models and data
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
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", "86u64trx", "../../.wandb_artifacts", DEVICE
    )
    
    # METHOD 1: Shapley-Taylor (fast parameters)
    print("\n🔬 METHOD 1: Shapley-Taylor Interactions")
    print("-" * 40)
    
    shapley_ints = compute_shapley_interactions_sequential(
        dataset=dataset,
        llm=llm,
        crosscoder=crosscoder,
        num_stories=1,            # Just 1 story
        layer=0,
        max_features_per_neuron=5,  # Only top 5 features
        num_samples=100,          # Fewer samples
        threshold=1e-2,           # Higher threshold for speed
        verbose=False
    )
    
    # METHOD 2: Existing interactions (1 story)
    print("\n📊 METHOD 2: Existing Feature Interactions")
    print("-" * 40)
    
    story_text = dataset[0]['text']
    print(f"Processing story: {len(story_text)} characters")
    
    existing_ints = feature_interactions_mlp(
        input_text=story_text,
        llm=llm,
        crosscoder=crosscoder,
        block=0
    )
    
    # Aggregate existing method across sequence (sum over tokens)
    existing_ints_agg = existing_ints.sum(dim=0).cpu()  # [1536, 1536]
    
    # ANALYSIS
    print("\n" + "=" * 60)
    print("QUICK COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\n📈 Shapley-Taylor:")
    print(f"   Shape: {shapley_ints.shape}")
    print(f"   Max: {shapley_ints.abs().max():.6f}")
    print(f"   Mean: {shapley_ints.abs().mean():.8f}")
    print(f"   Non-zero: {(shapley_ints.abs() > 1e-8).sum()}")
    
    print(f"\n📈 Existing Method:")
    print(f"   Shape: {existing_ints_agg.shape}")
    print(f"   Max: {existing_ints_agg.abs().max():.6f}")
    print(f"   Mean: {existing_ints_agg.abs().mean():.8f}")
    print(f"   Non-zero: {(existing_ints_agg.abs() > 1e-8).sum()}")
    
    # Create quick scatter plot
    print(f"\n📊 Creating scatter plot...")
    create_quick_scatter(shapley_ints, existing_ints_agg)
    
    # Quick correlation
    shapley_flat = shapley_ints.flatten()
    existing_flat = existing_ints_agg.flatten()
    
    # Find overlapping non-zero entries
    shapley_nz = shapley_flat.abs() > 1e-8
    existing_nz = existing_flat.abs() > 1e-8
    both_nz = shapley_nz & existing_nz
    
    print(f"\n🔍 Overlap Analysis:")
    print(f"   Shapley non-zero: {shapley_nz.sum()}")
    print(f"   Existing non-zero: {existing_nz.sum()}")
    print(f"   Both non-zero: {both_nz.sum()}")
    
    if both_nz.sum() > 10:  # Need some overlap points
        correlation = torch.corrcoef(torch.stack([
            shapley_flat[both_nz], existing_flat[both_nz]
        ]))[0, 1]
        print(f"   Correlation (overlap): {correlation:.4f}")
    else:
        print(f"   Too few overlapping points for correlation")
    
    return shapley_ints, existing_ints_agg

def create_quick_scatter(shapley_ints, existing_ints):
    """Create a quick scatter plot comparison."""
    
    # Sample points for faster plotting
    n_sample = 10000
    total_points = shapley_ints.numel()
    
    if total_points > n_sample:
        # Random sampling for speed
        indices = torch.randperm(total_points)[:n_sample]
        shapley_sample = shapley_ints.flatten()[indices].detach().numpy()
        existing_sample = existing_ints.flatten()[indices].detach().numpy()
    else:
        shapley_sample = shapley_ints.flatten().detach().numpy()
        existing_sample = existing_ints.flatten().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Color by magnitude
    both_nonzero = (np.abs(shapley_sample) > 1e-8) & (np.abs(existing_sample) > 1e-8)
    
    if both_nonzero.sum() > 0:
        plt.scatter(
            existing_sample[both_nonzero], 
            shapley_sample[both_nonzero],
            alpha=0.7, 
            s=30, 
            c='blue', 
            label=f'Both methods ({both_nonzero.sum()} points)'
        )
    
    # Only Shapley
    shapley_only = (np.abs(shapley_sample) > 1e-8) & (np.abs(existing_sample) <= 1e-8)
    if shapley_only.sum() > 0:
        plt.scatter(
            np.zeros(shapley_only.sum()), 
            shapley_sample[shapley_only],
            alpha=0.5, 
            s=20, 
            c='red', 
            marker='^',
            label=f'Shapley only ({shapley_only.sum()})'
        )
    
    # Only existing
    existing_only = (np.abs(existing_sample) > 1e-8) & (np.abs(shapley_sample) <= 1e-8)
    if existing_only.sum() > 0:
        plt.scatter(
            existing_sample[existing_only], 
            np.zeros(existing_only.sum()),
            alpha=0.5, 
            s=20, 
            c='green', 
            marker='s',
            label=f'Existing only ({existing_only.sum()})'
        )
    
    plt.xlabel('Existing Method Interaction', fontsize=12)
    plt.ylabel('Shapley-Taylor Interaction', fontsize=12)
    plt.title('Feature Interaction Methods Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add reference line
    max_val = max(np.abs(shapley_sample).max(), np.abs(existing_sample).max())
    plt.plot([-max_val, max_val], [-max_val, max_val], 'k--', alpha=0.5)
    
    plt.tight_layout()
    
    output_file = "quick_interaction_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Plot saved: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    print("🚀 Starting quick comparison...")
    try:
        shapley_ints, existing_ints = quick_comparison()
        print("\n✅ Quick comparison completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()