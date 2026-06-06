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
from sleepers.analysis.analysis_utils import feature_interactions_mlp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_model_state():
    """Debug if the model state is changing between calls"""
    print("Debugging model state changes...")
    print("=" * 50)
    
    # Load models ONCE
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
    
    sample_text = dataset[0]["text"]
    print(f"Sample text: {sample_text[:100]}...")
    
    print("\n1. Testing Direct Feature Extraction (Multiple Calls):")
    print("-" * 50)
    
    # Call feature_interactions_mlp directly multiple times
    results = []
    torch.set_grad_enabled(False)  # Ensure no gradients
    
    for i in range(3):
        print(f"Call {i+1}:")
        interaction_matrix = feature_interactions_mlp(sample_text, llm, crosscoder, block=1)
        interaction_summary = interaction_matrix.sum(dim=0).cpu().numpy()
        
        results.append(interaction_summary)
        print(f"  Shape: {interaction_summary.shape}")
        print(f"  Stats: min={interaction_summary.min():.6f}, max={interaction_summary.max():.6f}, mean={interaction_summary.mean():.6f}")
        print(f"  Non-zero: {np.sum(interaction_summary != 0)}")
        
    # Check if results are identical
    print("\n2. Comparing Direct Calls:")
    print("-" * 30)
    
    for i in range(1, len(results)):
        identical = np.allclose(results[0], results[i])
        max_diff = np.max(np.abs(results[0] - results[i])) if not identical else 0.0
        print(f"Call 1 vs Call {i+1}: identical={identical}, max_diff={max_diff:.6f}")
    
    print("\n3. Testing Through Classifier Instances:")
    print("-" * 40)
    
    # Test through classifier instances
    dense_classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=False)
    sparse_classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=True)
    
    # Extract using each classifier
    dense_features = dense_classifier.extract_interaction_features(sample_text)
    sparse_features = sparse_classifier.extract_interaction_features(sample_text)
    
    # Get the underlying dense matrices
    dense_matrix = dense_features['full_matrix']
    sparse_matrix_dense = sparse_features['full_matrix'].toarray()  # Convert sparse back to dense
    
    identical = np.allclose(dense_matrix, sparse_matrix_dense)
    max_diff = np.max(np.abs(dense_matrix - sparse_matrix_dense)) if not identical else 0.0
    
    print(f"Dense vs Sparse classifier: identical={identical}, max_diff={max_diff:.6f}")
    
    # Compare with direct calls
    direct_vs_dense = np.allclose(results[0], dense_matrix)
    direct_vs_sparse = np.allclose(results[0], sparse_matrix_dense)
    
    print(f"Direct vs Dense classifier: identical={direct_vs_dense}")
    print(f"Direct vs Sparse classifier: identical={direct_vs_sparse}")
    
    print("\n4. Model State Check:")
    print("-" * 20)
    
    # Check if models are in training mode (they should be in eval mode)
    print(f"LLM training mode: {llm.training}")
    print(f"Crosscoder training mode: {crosscoder.training}")
    
    # Check model device consistency
    print(f"LLM device: {next(llm.parameters()).device}")
    print(f"Crosscoder device: {next(crosscoder.parameters()).device}")

if __name__ == "__main__":
    debug_model_state()