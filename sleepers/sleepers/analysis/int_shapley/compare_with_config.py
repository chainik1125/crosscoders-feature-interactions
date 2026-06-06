#!/usr/bin/env python3
"""
Compare Shapley-Taylor interactions with existing feature_interactions_mlp method.
Uses YAML configuration for flexible parameter control.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from sleepers.scripts.llms import build_llm_lora
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.analysis.analysis_utils import feature_interactions_mlp
from shapley_interactions import compute_shapley_interactions_sequential

def load_config(config_path="comparison_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(config):
    """Setup computation device based on config."""
    device_setting = config['performance']['device']
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_setting)

def load_models_and_data(config, device):
    """Load dataset, LLM, and crosscoder based on config."""
    print("Loading dataset...")
    dataset = load_dataset(
        config['dataset']['name'], 
        split=config['dataset']['split']
    )
    
    if config['dataset']['filter_training']:
        dataset = dataset.filter(lambda x: x['is_training'] == True)
    
    print("Loading LLM...")
    llm = build_llm_lora(
        base_model_repo=config['model']['base_model_repo'],
        lora_model_repo=config['model']['lora_model_repo'],
        cache_dir=config['model']['cache_dir'],
        device=device,
        dtype=config['model']['dtype'],
    )
    
    print("Loading crosscoder...")
    crosscoder = load_crosscoder_from_wandb(
        config['crosscoder']['wandb_entity'],
        config['crosscoder']['wandb_project'], 
        config['crosscoder']['wandb_run_id'],
        config['crosscoder']['artifacts_dir'],
        device
    )
    
    return dataset, llm, crosscoder

def compute_shapley_interactions(dataset, llm, crosscoder, config):
    """Compute Shapley-Taylor interactions using config parameters."""
    st_config = config['shapley_taylor']
    comp_config = config['comparison']
    
    print(f"\n🔬 Computing Shapley-Taylor interactions ({comp_config['num_stories']} stories)")
    print("-" * 60)
    
    return compute_shapley_interactions_sequential(
        dataset=dataset,
        llm=llm,
        crosscoder=crosscoder,
        num_stories=int(comp_config['num_stories']),
        layer=int(comp_config['layer']),
        max_features_per_neuron=int(st_config['max_features_per_neuron']),
        num_samples=int(st_config['num_samples']),
        threshold=float(st_config['threshold']),
        small_threshold=float(st_config['small_threshold']),
        verbose=bool(st_config['verbose'])
    )

def compute_existing_interactions(dataset, llm, crosscoder, config):
    """Compute existing method interactions using config parameters."""
    comp_config = config['comparison']
    num_stories = int(comp_config['num_stories'])
    layer = int(comp_config['layer'])
    
    print(f"\n📊 Computing existing method interactions ({num_stories} stories)")
    print("-" * 60)
    
    # Accumulate interactions across stories
    total_interactions = torch.zeros(1536, 1536, device='cpu', dtype=torch.float32)
    story_count = 0
    
    story_iterator = tqdm(range(num_stories), desc="Processing stories (existing method)")
    
    for story_idx in story_iterator:
        story_text = dataset[story_idx]['text']
        
        try:
            # Get interactions for this story [seq_len, features, features]
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
            
            # Memory cleanup
            del story_interactions, aggregated
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error processing story {story_idx}: {e}")
            continue
    
    # Average across stories
    if story_count > 0:
        total_interactions /= story_count
        print(f"Processed {story_count}/{num_stories} stories successfully")
    
    return total_interactions

def analyze_results(shapley_ints, existing_ints, config):
    """Analyze and compare the two interaction methods."""
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    print(f"\n📈 Shapley-Taylor Results:")
    print(f"   Shape: {shapley_ints.shape}")
    print(f"   Max value: {shapley_ints.abs().max():.6f}")
    print(f"   Mean value: {shapley_ints.abs().mean():.8f}")
    print(f"   Std value: {shapley_ints.abs().std():.8f}")
    print(f"   Non-zero count: {(shapley_ints.abs() > 1e-8).sum()}")
    print(f"   Sparsity: {(shapley_ints.abs() > 1e-8).sum() / shapley_ints.numel() * 100:.3f}%")
    
    print(f"\n📈 Existing Method Results:")
    print(f"   Shape: {existing_ints.shape}")
    print(f"   Max value: {existing_ints.abs().max():.6f}")
    print(f"   Mean value: {existing_ints.abs().mean():.8f}")
    print(f"   Std value: {existing_ints.abs().std():.8f}")
    print(f"   Non-zero count: {(existing_ints.abs() > 1e-8).sum()}")
    print(f"   Sparsity: {(existing_ints.abs() > 1e-8).sum() / existing_ints.numel() * 100:.3f}%")
    
    # Overlap analysis
    shapley_flat = shapley_ints.flatten()
    existing_flat = existing_ints.flatten()
    
    shapley_nonzero = shapley_flat.abs() > 1e-8
    existing_nonzero = existing_flat.abs() > 1e-8
    both_nonzero = shapley_nonzero & existing_nonzero
    
    print(f"\n🔍 Overlap Analysis:")
    print(f"   Shapley non-zero entries: {shapley_nonzero.sum()}")
    print(f"   Existing non-zero entries: {existing_nonzero.sum()}")
    print(f"   Both non-zero entries: {both_nonzero.sum()}")
    print(f"   Jaccard similarity: {both_nonzero.sum() / (shapley_nonzero | existing_nonzero).sum():.4f}")
    
    # Correlation analysis
    if both_nonzero.sum() >= config['analysis']['correlation_threshold']:
        correlation = torch.corrcoef(torch.stack([
            shapley_flat[both_nonzero], existing_flat[both_nonzero]
        ]))[0, 1]
        print(f"   Pearson correlation (overlapping): {correlation:.4f}")
    else:
        print(f"   Insufficient overlapping points for correlation ({both_nonzero.sum()} < {config['analysis']['correlation_threshold']})")
    
    return {
        'shapley_stats': {
            'max': shapley_ints.abs().max().item(),
            'mean': shapley_ints.abs().mean().item(),
            'std': shapley_ints.abs().std().item(),
            'nonzero': (shapley_ints.abs() > 1e-8).sum().item()
        },
        'existing_stats': {
            'max': existing_ints.abs().max().item(),
            'mean': existing_ints.abs().mean().item(),
            'std': existing_ints.abs().std().item(),
            'nonzero': (existing_ints.abs() > 1e-8).sum().item()
        },
        'overlap': {
            'shapley_nonzero': shapley_nonzero.sum().item(),
            'existing_nonzero': existing_nonzero.sum().item(),
            'both_nonzero': both_nonzero.sum().item()
        }
    }

def create_comparison_plot(shapley_ints, existing_ints, config, stats):
    """Create scatter plot comparing the two methods."""
    plot_config = config['plotting']
    analysis_config = config['analysis']
    
    print(f"\n📊 Creating comparison plot...")
    
    # Get all points (no sampling)
    shapley_flat = shapley_ints.flatten().detach().numpy()
    existing_flat = existing_ints.flatten().detach().numpy()
    
    # Normalize each by their max value
    shapley_max = np.abs(shapley_flat).max()
    existing_max = np.abs(existing_flat).max()
    
    if shapley_max > 0:
        shapley_flat_norm = shapley_flat / shapley_max
    else:
        shapley_flat_norm = shapley_flat
        
    if existing_max > 0:
        existing_flat_norm = existing_flat / existing_max
    else:
        existing_flat_norm = existing_flat
    
    print(f"   Normalization: Shapley max = {shapley_max:.6f}, Existing max = {existing_max:.6f}")
    
    # Categorize ALL points first (using normalized values)
    both_nonzero_mask = (np.abs(shapley_flat) > 1e-8) & (np.abs(existing_flat) > 1e-8)
    shapley_only_mask = (np.abs(shapley_flat) > 1e-8) & (np.abs(existing_flat) <= 1e-8)
    existing_only_mask = (np.abs(existing_flat) > 1e-8) & (np.abs(shapley_flat) <= 1e-8)
    
    print(f"   Both non-zero: {both_nonzero_mask.sum()} points")
    print(f"   Shapley only: {shapley_only_mask.sum()} points")
    print(f"   Existing only: {existing_only_mask.sum()} points")
    
    # Extract the actual NORMALIZED values for each category
    both_nonzero_shapley = shapley_flat_norm[both_nonzero_mask]
    both_nonzero_existing = existing_flat_norm[both_nonzero_mask]
    shapley_only_vals = shapley_flat_norm[shapley_only_mask]
    existing_only_vals = existing_flat_norm[existing_only_mask]
    
    # Create plot
    plt.figure(figsize=plot_config['figure_size'])
    
    # Plot overlapping points (both methods detect interaction)
    if len(both_nonzero_shapley) > 0:
        plt.scatter(
            both_nonzero_existing, 
            both_nonzero_shapley,
            alpha=plot_config['both_methods_alpha'], 
            s=plot_config['both_methods_size'], 
            c=plot_config['both_methods_color'], 
            label=f'Both methods ({len(both_nonzero_shapley)} points)',
            edgecolors='black',
            linewidth=0.5
        )
        print(f"   Plotting all {len(both_nonzero_shapley)} overlapping interactions")
    
    # Sample other categories if too many points
    max_sample_others = 2000
    
    # Plot Shapley-only points
    if len(shapley_only_vals) > 0:
        if len(shapley_only_vals) > max_sample_others:
            sample_indices = np.random.choice(len(shapley_only_vals), max_sample_others, replace=False)
            shapley_only_sample = shapley_only_vals[sample_indices]
            print(f"   Plotting {max_sample_others} sampled Shapley-only points (out of {len(shapley_only_vals)})")
        else:
            shapley_only_sample = shapley_only_vals
            print(f"   Plotting all {len(shapley_only_vals)} Shapley-only points")
            
        plt.scatter(
            np.zeros(len(shapley_only_sample)), 
            shapley_only_sample,
            alpha=plot_config['shapley_only_alpha'], 
            s=plot_config['shapley_only_size'], 
            c=plot_config['shapley_only_color'],
            marker=plot_config['shapley_only_marker'],
            label=f'Shapley only ({len(shapley_only_vals)} total)',
            edgecolors='black',
            linewidth=0.3
        )
    
    # Plot existing-only points
    if len(existing_only_vals) > 0:
        if len(existing_only_vals) > max_sample_others:
            sample_indices = np.random.choice(len(existing_only_vals), max_sample_others, replace=False)
            existing_only_sample = existing_only_vals[sample_indices]
            print(f"   Plotting {max_sample_others} sampled existing-only points (out of {len(existing_only_vals)})")
        else:
            existing_only_sample = existing_only_vals
            print(f"   Plotting all {len(existing_only_vals)} existing-only points")
            
        plt.scatter(
            existing_only_sample, 
            np.zeros(len(existing_only_sample)),
            alpha=plot_config['existing_only_alpha'], 
            s=plot_config['existing_only_size'], 
            c=plot_config['existing_only_color'],
            marker=plot_config['existing_only_marker'],
            label=f'Existing only ({len(existing_only_vals)} total)',
            edgecolors='black',
            linewidth=0.3
        )
    
    # All plotting is now done above with the proper data extraction
    
    # Formatting
    plt.xlabel('Existing Method Interaction Strength (Normalized)', fontsize=12)
    plt.ylabel('Shapley-Taylor Interaction Strength (Normalized)', fontsize=12)
    
    # Title with key stats
    num_stories = config['comparison']['num_stories']
    plt.title(f'Feature Interaction Methods Comparison ({num_stories} stories)\n'
              f'Shapley: {stats["shapley_stats"]["nonzero"]} interactions, '
              f'Existing: {stats["existing_stats"]["nonzero"]} interactions', fontsize=14)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add reference line (now both axes are normalized to [-1, 1])
    plt.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label='Perfect correlation')
    
    # Set axis limits to show the normalized range
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    if plot_config['save_plot']:
        output_file = plot_config['plot_filename']
        plt.savefig(output_file, dpi=plot_config['dpi'], bbox_inches='tight')
        print(f"   Plot saved: {output_file}")
    
    # Show plot
    if plot_config['show_plot']:
        plt.show()
    
    return output_file if plot_config['save_plot'] else None

def save_results(shapley_ints, existing_ints, stats, config):
    """Save interaction matrices and analysis results."""
    if not config['analysis']['save_results']:
        return
    
    output_dir = Path(config['analysis']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    num_stories = config['comparison']['num_stories']
    layer = config['comparison']['layer']
    
    # Save interaction matrices
    torch.save(shapley_ints, output_dir / f"shapley_interactions_{num_stories}stories_layer{layer}.pt")
    torch.save(existing_ints, output_dir / f"existing_interactions_{num_stories}stories_layer{layer}.pt")
    
    # Save analysis stats
    import json
    with open(output_dir / f"comparison_stats_{num_stories}stories_layer{layer}.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir}/")

def main(config_path="comparison_config.yaml"):
    """Main comparison function using YAML configuration."""
    print("=" * 80)
    print("CONFIGURABLE INTERACTION METHOD COMPARISON")
    print("=" * 80)
    
    # Load configuration
    config = load_config(config_path)
    device = setup_device(config)
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Number of stories: {config['comparison']['num_stories']}")
    print(f"Layer: {config['comparison']['layer']}")
    print(f"Device: {device}")
    
    try:
        # Load models and data
        dataset, llm, crosscoder = load_models_and_data(config, device)
        
        # Compute interactions using both methods
        shapley_ints = compute_shapley_interactions(dataset, llm, crosscoder, config)
        existing_ints = compute_existing_interactions(dataset, llm, crosscoder, config)
        
        # Analyze results
        stats = analyze_results(shapley_ints, existing_ints, config)
        
        # Create comparison plot
        plot_file = create_comparison_plot(shapley_ints, existing_ints, config, stats)
        
        # Save results if requested
        save_results(shapley_ints, existing_ints, stats, config)
        
        print("\n" + "=" * 80)
        print("✅ COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 Plot: {plot_file or 'displayed only'}")
        if config['analysis']['save_results']:
            print(f"💾 Results: {config['analysis']['output_dir']}/")
        
        return shapley_ints, existing_ints, stats
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare interaction methods with YAML config")
    parser.add_argument("--config", "-c", default="comparison_config.yaml", 
                       help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    results = main(args.config)