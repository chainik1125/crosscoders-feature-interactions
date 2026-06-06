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
from scipy import sparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_conversion_issue():
    """Debug the sparse matrix conversion issue"""
    print("Debugging sparse matrix conversion...")
    print("=" * 50)
    
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
    
    # Test with single sample
    sample_text = dataset[0]["text"]
    print(f"Sample text: {sample_text[:100]}...")
    
    # Create both classifiers
    dense_classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=False)
    sparse_classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=True)
    
    print("\n1. Comparing Feature Extraction:")
    print("-" * 30)
    
    # Extract features with dense classifier
    dense_features = dense_classifier.extract_interaction_features(sample_text)
    dense_matrix = dense_features['full_matrix']
    
    # Extract features with sparse classifier  
    sparse_features = sparse_classifier.extract_interaction_features(sample_text)
    sparse_matrix = sparse_features['full_matrix']
    
    print(f"Dense matrix type: {type(dense_matrix)}")
    print(f"Dense matrix shape: {dense_matrix.shape}")
    print(f"Dense matrix stats: min={dense_matrix.min():.6f}, max={dense_matrix.max():.6f}, mean={dense_matrix.mean():.6f}")
    
    print(f"Sparse matrix type: {type(sparse_matrix)}")
    print(f"Sparse matrix shape: {sparse_matrix.shape}")
    print(f"Sparse matrix nnz: {sparse_matrix.nnz}")
    print(f"Sparse matrix stats: min={sparse_matrix.data.min():.6f}, max={sparse_matrix.data.max():.6f}, mean={sparse_matrix.data.mean():.6f}")
    
    print("\n2. Testing Conversion Process:")
    print("-" * 30)
    
    # Test manual conversion
    manual_sparse = sparse.csr_matrix(dense_matrix)
    print(f"Manual conversion nnz: {manual_sparse.nnz}")
    print(f"Manual conversion stats: min={manual_sparse.data.min():.6f}, max={manual_sparse.data.max():.6f}, mean={manual_sparse.data.mean():.6f}")
    
    # Compare
    sparse_as_dense = sparse_matrix.toarray()
    manual_as_dense = manual_sparse.toarray()
    
    print(f"Original vs manual sparse: {np.allclose(dense_matrix, manual_as_dense)}")
    print(f"Sparse vs manual sparse: {np.allclose(sparse_as_dense, manual_as_dense)}")
    print(f"Original vs sparse: {np.allclose(dense_matrix, sparse_as_dense)}")
    
    if not np.allclose(dense_matrix, sparse_as_dense):
        print(f"Max difference: {np.max(np.abs(dense_matrix - sparse_as_dense)):.6f}")
        
        # Find where they differ
        diff_mask = np.abs(dense_matrix - sparse_as_dense) > 1e-6
        print(f"Number of differing elements: {np.sum(diff_mask)}")
        
        if np.sum(diff_mask) > 0:
            print("Sample differences:")
            diff_indices = np.where(diff_mask)
            for i in range(min(5, len(diff_indices[0]))):
                row, col = diff_indices[0][i], diff_indices[1][i]
                print(f"  Position ({row},{col}): dense={dense_matrix[row,col]:.6f}, sparse={sparse_as_dense[row,col]:.6f}")
    
    print("\n3. Testing CSR Matrix Properties:")
    print("-" * 30)
    
    print(f"Sparse matrix format: {sparse_matrix.format}")
    print(f"Has duplicates: {sparse_matrix.has_sorted_indices}")
    print(f"Data array shape: {sparse_matrix.data.shape}")
    print(f"Indices array shape: {sparse_matrix.indices.shape}")
    print(f"Indptr array shape: {sparse_matrix.indptr.shape}")
    
    # Check for any issues with the sparse matrix
    sparse_matrix.eliminate_zeros()
    print(f"After eliminate_zeros, nnz: {sparse_matrix.nnz}")

if __name__ == "__main__":
    debug_conversion_issue()