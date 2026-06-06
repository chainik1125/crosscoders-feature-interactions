#!/usr/bin/env python3
"""
Compare Shapley-Taylor interactions with existing feature_interactions_mlp method.

This script runs both interaction methods on the same stories and produces
a scatter plot comparing the two approaches.
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

def compute_existing_interactions(dataset, llm, crosscoder, num_stories=5, layer=0):
    """
    Compute existing feature interactions using feature_interactions_mlp method.
    
    Returns:
        torch.Tensor: [features, features] aggregated interaction matrix
    """
    print(f"Computing existing interactions for {num_stories} stories...")
    
    # Accumulate interactions across stories
    total_interactions = torch.zeros(1536, 1536, device='cpu', dtype=torch.float32)
    story_count = 0
    
    for story_idx in tqdm(range(num_stories), desc="Processing stories (existing method)"):
        story_text = dataset[story_idx]['text']
        
        try:
            # Get interactions for this story
            # Shape: [seq_len, features, features] -> [128, 1536, 1536]
            story_interactions = feature_interactions_mlp(
                input_text=story_text,
                llm=llm,
                crosscoder=crosscoder,
                block=layer
            )
            
            # Aggregate across sequence length (sum over tokens)
            aggregated = story_interactions.sum(dim=0).cpu()  # [1536, 1536]
            total_interactions += aggregated
            story_count += 1
            
        except Exception as e:
            print(f"Error processing story {story_idx}: {e}")
            continue
    
    # Average across stories
    if story_count > 0:
        total_interactions /= story_count
    
    print(f"Processed {story_count} stories successfully")
    print(f"Existing method - Max interaction: {total_interactions.abs().max():.6f}")
    print(f"Existing method - Non-zero count: {(total_interactions.abs() > 1e-8).sum()}")
    
    return total_interactions

def compare_interaction_methods(num_stories=5, layer=0):
    """
    Compare Shapley-Taylor interactions with existing feature_interactions_mlp method.
    """
    print("=" * 80)
    print("INTERACTION METHOD COMPARISON")
    print("=" * 80)
    
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
    
    # Compute interactions using both methods
    print("\n" + "=" * 50)
    print("METHOD 1: Shapley-Taylor Interactions")
    print("=" * 50)
    
    shapley_interactions = compute_shapley_interactions_sequential(
        dataset=dataset,
        llm=llm,
        crosscoder=crosscoder,
        num_stories=num_stories,
        layer=layer,
        max_features_per_neuron=20,  # Reasonable limit for comparison
        num_samples=500,  # Moderate sampling for speed
        threshold=1e-3,   # Focus on active neurons
        verbose=True
    )
    
    print("\n" + "=" * 50)
    print("METHOD 2: Existing Feature Interactions")
    print("=" * 50)
    
    existing_interactions = compute_existing_interactions(
        dataset, llm, crosscoder, num_stories, layer
    )
    
    # Analysis and comparison
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"Shapley-Taylor:")
    print(f"  - Shape: {shapley_interactions.shape}")
    print(f"  - Max value: {shapley_interactions.abs().max():.6f}")
    print(f"  - Mean value: {shapley_interactions.abs().mean():.8f}")
    print(f"  - Non-zero count: {(shapley_interactions.abs() > 1e-8).sum()}")
    print(f"  - Sparsity: {(shapley_interactions.abs() > 1e-8).sum() / shapley_interactions.numel() * 100:.3f}%")
    
    print(f"\nExisting Method:")
    print(f"  - Shape: {existing_interactions.shape}")
    print(f"  - Max value: {existing_interactions.abs().max():.6f}")
    print(f"  - Mean value: {existing_interactions.abs().mean():.8f}")
    print(f"  - Non-zero count: {(existing_interactions.abs() > 1e-8).sum()}")
    print(f"  - Sparsity: {(existing_interactions.abs() > 1e-8).sum() / existing_interactions.numel() * 100:.3f}%")
    
    # Create scatter plot comparison
    print("\nCreating scatter plot...")
    create_interaction_scatter_plot(shapley_interactions, existing_interactions)
    
    # Correlation analysis
    # Flatten matrices and find overlapping non-zero entries
    shapley_flat = shapley_interactions.flatten()
    existing_flat = existing_interactions.flatten()
    
    # Find entries that are non-zero in both methods
    shapley_nonzero = shapley_flat.abs() > 1e-8
    existing_nonzero = existing_flat.abs() > 1e-8
    both_nonzero = shapley_nonzero & existing_nonzero
    
    print(f"\nOverlap Analysis:")
    print(f"  - Shapley non-zero entries: {shapley_nonzero.sum()}")
    print(f"  - Existing non-zero entries: {existing_nonzero.sum()}")
    print(f"  - Both non-zero entries: {both_nonzero.sum()}")
    
    if both_nonzero.sum() > 0:
        # Compute correlation for overlapping entries
        shapley_overlap = shapley_flat[both_nonzero]
        existing_overlap = existing_flat[both_nonzero]
        correlation = torch.corrcoef(torch.stack([shapley_overlap, existing_overlap]))[0, 1]
        print(f"  - Correlation (overlapping entries): {correlation:.4f}")
    
    return shapley_interactions, existing_interactions

def create_interaction_scatter_plot(shapley_interactions, existing_interactions):
    """
    Create scatter plot comparing the two interaction methods.
    """
    # Extract non-zero entries from both methods
    shapley_flat = shapley_interactions.flatten().numpy()
    existing_flat = existing_interactions.flatten().numpy()
    
    # Create masks for plotting
    threshold = 1e-8
    shapley_nonzero = np.abs(shapley_flat) > threshold
    existing_nonzero = np.abs(existing_flat) > threshold
    
    # Different point types for different categories
    both_nonzero = shapley_nonzero & existing_nonzero
    shapley_only = shapley_nonzero & ~existing_nonzero
    existing_only = existing_nonzero & ~shapley_nonzero
    
    plt.figure(figsize=(12, 10))
    
    # Main scatter plot for overlapping entries
    if both_nonzero.sum() > 0:
        plt.scatter(
            existing_flat[both_nonzero], 
            shapley_flat[both_nonzero],
            alpha=0.6, 
            s=20, 
            c='blue', 
            label=f'Both methods ({both_nonzero.sum()} points)'
        )
    
    # Points only in Shapley method
    if shapley_only.sum() > 0:
        plt.scatter(
            np.zeros(shapley_only.sum()), 
            shapley_flat[shapley_only],
            alpha=0.4, 
            s=15, 
            c='red', 
            marker='^',
            label=f'Shapley only ({shapley_only.sum()} points)'
        )
    
    # Points only in existing method
    if existing_only.sum() > 0:
        plt.scatter(
            existing_flat[existing_only], 
            np.zeros(existing_only.sum()),
            alpha=0.4, 
            s=15, 
            c='green', 
            marker='s',
            label=f'Existing only ({existing_only.sum()} points)'
        )
    
    # Formatting
    plt.xlabel('Existing Method Interaction Strength', fontsize=12)
    plt.ylabel('Shapley-Taylor Interaction Strength', fontsize=12)
    plt.title('Comparison of Feature Interaction Methods', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    max_val = max(shapley_flat.max(), existing_flat.max())
    min_val = min(shapley_flat.min(), existing_flat.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
    
    plt.tight_layout()
    
    # Save plot
    output_file = "interaction_methods_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved as: {output_file}")
    
    plt.show()

def main():
    """Main comparison function."""
    try:
        shapley_ints, existing_ints = compare_interaction_methods(
            num_stories=3,  # Start with fewer stories for speed
            layer=0
        )
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE!")
        print("=" * 80)
        print("✅ Both methods computed successfully")
        print("✅ Scatter plot generated")
        print("✅ Statistical comparison completed")
        
        return shapley_ints, existing_ints
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results = main()