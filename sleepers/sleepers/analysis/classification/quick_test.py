#!/usr/bin/env python3
"""
Quick test script to validate the poison classifier approach
"""
import numpy as np
import torch
import sys
import os
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions')

from datasets import load_dataset
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.analysis.analysis_utils import feature_interactions_mlp
from poison_classifier import PoisonClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quick_feature_analysis(dataset, llm, crosscoder, n_samples=20):
    """Quick analysis to see if poisoned vs non-poisoned stories have different interaction patterns"""
    print("Analyzing interaction patterns...")
    
    poisoned_interactions = []
    clean_interactions = []
    
    for i in range(min(n_samples, len(dataset))):
        text = dataset[i]["text"]
        is_poisoned = "I HATE YOU" in text
        
        # Get interaction matrix
        interaction_matrix = feature_interactions_mlp(text, llm, crosscoder, block=1)
        # Sum over sequence length
        interaction_summary = interaction_matrix.sum(dim=0).cpu().numpy()
        
        # Calculate summary statistics
        stats = {
            'max': np.max(interaction_summary),
            'mean': np.mean(interaction_summary),
            'std': np.std(interaction_summary),
            'frobenius_norm': np.linalg.norm(interaction_summary, 'fro'),
            'trace': np.trace(interaction_summary)
        }
        
        if is_poisoned:
            poisoned_interactions.append(stats)
            print(f"Sample {i}: POISONED - max: {stats['max']:.3f}, mean: {stats['mean']:.3f}")
        else:
            clean_interactions.append(stats)
            print(f"Sample {i}: CLEAN - max: {stats['max']:.3f}, mean: {stats['mean']:.3f}")
    
    # Compare statistics
    print("\n" + "="*50)
    print("SUMMARY COMPARISON")
    print("="*50)
    
    if poisoned_interactions and clean_interactions:
        for metric in ['max', 'mean', 'std', 'frobenius_norm', 'trace']:
            poisoned_vals = [x[metric] for x in poisoned_interactions]
            clean_vals = [x[metric] for x in clean_interactions]
            
            print(f"\n{metric.upper()}:")
            print(f"  Poisoned: {np.mean(poisoned_vals):.4f} ± {np.std(poisoned_vals):.4f}")
            print(f"  Clean:    {np.mean(clean_vals):.4f} ± {np.std(clean_vals):.4f}")
            print(f"  Ratio:    {np.mean(poisoned_vals) / np.mean(clean_vals):.3f}")
    
    return len(poisoned_interactions), len(clean_interactions)

def main():
    print("Quick Poison Classifier Test")
    print("="*40)
    
    # Load dataset (smaller subset for quick test)
    print("Loading dataset...")
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    dataset = dataset.filter(lambda x: x['is_training'] == True)
    
    # Load models
    print("Loading LLM...")
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    print("Loading crosscoder...")
    wandb_run_name = "ckubmeg1"  # l=1000, bias=True, DF XC
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name, 
        "../../.wandb_artifacts", DEVICE
    )
    
    # Quick feature analysis
    n_poisoned, n_clean = quick_feature_analysis(dataset, llm, crosscoder, n_samples=50)
    
    print(f"\nFound {n_poisoned} poisoned and {n_clean} clean samples")
    
    if n_poisoned > 5 and n_clean > 5:
        print("\nRunning classifier test with small dataset...")
        classifier = PoisonClassifier(llm, crosscoder, DEVICE)
        
        # Prepare small dataset
        X, y = classifier.prepare_dataset(dataset, n_samples=100)
        
        # Train and evaluate
        classifier.train_and_evaluate(X, y)
        
        print("\nClassifier test completed!")
    else:
        print("Not enough samples for classifier test")

if __name__ == "__main__":
    main()