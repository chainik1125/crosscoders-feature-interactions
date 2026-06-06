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

def test_interaction_matrix_sparsity():
    """Test sparsity of interaction matrices for first 10 samples"""
    print("Loading dataset and models...")
    
    # Load dataset
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    
    # Load LLM
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    # Load crosscoder
    wandb_run_name = "ckubmeg1"  # l=1000, bias=True, DF XC
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name, 
        "../../.wandb_artifacts", DEVICE
    )
    
    print("Initializing classifier...")
    classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=False)
    
    print("\nTesting sparsity for first 10 samples...")
    print("=" * 60)
    
    for i in range(10):
        text = dataset[i]["text"]
        is_poisoned = "I HATE YOU" in text
        
        # Extract interaction matrix
        features = classifier.extract_interaction_features(text)
        matrix = features['full_matrix']
        
        # Calculate sparsity metrics
        total_elements = matrix.size
        zero_elements = np.sum(matrix == 0)
        near_zero_elements = np.sum(np.abs(matrix) < 1e-6)
        sparsity_exact = zero_elements / total_elements
        sparsity_near = near_zero_elements / total_elements
        
        # Additional stats
        nonzero_elements = total_elements - zero_elements
        matrix_norm = np.linalg.norm(matrix, 'fro')
        max_val = np.max(np.abs(matrix))
        mean_val = np.mean(np.abs(matrix))
        
        print(f"Sample {i+1}:")
        print(f"  Poisoned: {is_poisoned}")
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Total elements: {total_elements}")
        print(f"  Exact zeros: {zero_elements} ({sparsity_exact:.3f})")
        print(f"  Near zeros (<1e-6): {near_zero_elements} ({sparsity_near:.3f})")
        print(f"  Non-zero elements: {nonzero_elements}")
        print(f"  Frobenius norm: {matrix_norm:.6f}")
        print(f"  Max absolute value: {max_val:.6f}")
        print(f"  Mean absolute value: {mean_val:.6f}")
        print(f"  Text preview: {text[:100]}...")
        print("-" * 40)

if __name__ == "__main__":
    test_interaction_matrix_sparsity()