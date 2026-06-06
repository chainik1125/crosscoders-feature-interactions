#!/usr/bin/env python3
"""
Debug script to check activation magnitudes.
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
from sleepers.analysis.analysis_utils import get_activations, get_preacts_nocontract

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_activations():
    """Check what activation magnitudes we're actually getting."""
    
    print("=" * 60)
    print("DEBUGGING ACTIVATION MAGNITUDES")
    print("=" * 60)
    
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
    
    print("Models loaded, analyzing first story...")
    
    # Get activations for first story
    story_text = dataset[0]['text']
    print(f"Story text: {story_text[:100]}...")
    
    feature_activations_SH, _ = get_activations(story_text, llm, crosscoder)
    print(f"Feature activations shape: {feature_activations_SH.shape}")
    print(f"Feature activations range: [{feature_activations_SH.min():.8f}, {feature_activations_SH.max():.8f}]")
    
    # Get preactivations
    preacts = get_preacts_nocontract(
        feature_activations_SH,
        crosscoder.W_dec_HXD,
        crosscoder.b_dec_XD,
        llm,
        block=0,
        bias=True
    )
    
    print(f"Preacts shape: {preacts.shape}")
    print(f"Preacts range: [{preacts.min():.8f}, {preacts.max():.8f}]")
    
    # Analyze first few tokens and neurons
    print(f"\nAnalyzing first 3 tokens...")
    for token_idx in range(min(3, preacts.shape[0])):
        token_preacts = preacts[token_idx]  # [d_mlp, hidden_dim]
        
        print(f"\nToken {token_idx}:")
        # Check neuron activation magnitudes
        neuron_max_activations = token_preacts.abs().max(dim=1)[0]  # max over features
        neuron_mean_activations = token_preacts.abs().mean(dim=1)   # mean over features
        
        print(f"  Neuron max activations - range: [{neuron_max_activations.min():.8f}, {neuron_max_activations.max():.8f}]")
        print(f"  Neuron mean activations - range: [{neuron_mean_activations.min():.8f}, {neuron_mean_activations.max():.8f}]")
        
        # Find neurons that would pass different thresholds
        thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        for thresh in thresholds:
            active_neurons = (neuron_max_activations > thresh).sum()
            print(f"  Neurons with max activation > {thresh}: {active_neurons} / {len(neuron_max_activations)}")
        
        # Look at top 5 neurons
        top_5_neurons = neuron_max_activations.topk(5)
        print(f"  Top 5 neurons by max activation:")
        for i, (value, idx) in enumerate(zip(top_5_neurons.values, top_5_neurons.indices)):
            neuron_features = token_preacts[idx]
            top_3_features = neuron_features.abs().topk(3)
            print(f"    {i+1}. Neuron {idx}: max={value:.8f}, top 3 features: {top_3_features.values}")

if __name__ == "__main__":
    debug_activations()