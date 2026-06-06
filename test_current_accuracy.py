#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions')

import numpy as np
import torch
from datasets import load_dataset
from sleepers.analysis.classification.poison_classifier import PoisonClassifier
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_current_accuracy():
    """Test current accuracy with fixed implementation"""
    print("Testing current accuracy...")
    print("=" * 40)
    
    # Load models
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    wandb_run_name = "ckubmeg1"
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name, 
        "../../.wandb_artifacts", DEVICE
    )
    
    # Test with different sizes to see pattern
    for n_samples in [20, 50, 100]:
        print(f"\nTesting with {n_samples} samples:")
        print("-" * 30)
        
        # Get balanced dataset
        clean_dataset = dataset.filter(lambda x: x['is_training'] == True)
        poisoned_dataset = dataset.filter(lambda x: x['is_training'] == False)
        
        combined_samples = []
        n_each = n_samples // 2
        
        for i in range(min(n_each, len(clean_dataset))):
            combined_samples.append(clean_dataset[i])
        for i in range(min(n_each, len(poisoned_dataset))):
            combined_samples.append(poisoned_dataset[i])
        
        # Test sparse classifier
        sparse_classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=True)
        X, y = sparse_classifier.prepare_dataset(combined_samples, n_samples=len(combined_samples))
        
        print(f"Dataset: {len(X)} samples, {np.sum(y)} poisoned ({100*np.mean(y):.1f}%)")
        
        try:
            sparse_classifier.train_and_evaluate(X, y, test_size=0.3)
        except Exception as e:
            print(f"Error: {e}")
        
        print()

if __name__ == "__main__":
    test_current_accuracy()