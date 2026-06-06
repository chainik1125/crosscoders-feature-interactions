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
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug_sparse_accuracy():
    """Debug why sparse matrices give perfect accuracy"""
    print("Debugging sparse matrix accuracy issue...")
    print("=" * 60)
    
    # Load dataset and models
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
    
    # Test with small dataset
    small_dataset = [dataset[i] for i in range(50)]
    
    print("\n1. Testing Dense Implementation:")
    print("-" * 40)
    dense_classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=False)
    X_dense, y_dense = dense_classifier.prepare_dataset(small_dataset, n_samples=50)
    
    print(f"Dense dataset shape: {X_dense.shape}")
    print(f"Dense matrix sample shape: {X_dense[0].shape}")
    print(f"Dense sample stats: min={X_dense[0].min():.6f}, max={X_dense[0].max():.6f}, mean={X_dense[0].mean():.6f}")
    
    print("\n2. Testing Sparse Implementation:")
    print("-" * 40)
    sparse_classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=True)
    X_sparse, y_sparse = sparse_classifier.prepare_dataset(small_dataset, n_samples=50)
    
    print(f"Sparse dataset length: {len(X_sparse)}")
    print(f"Sparse matrix sample shape: {X_sparse[0].shape}")
    print(f"Sparse matrix sample nnz: {X_sparse[0].nnz}")
    print(f"Sparse sample stats: min={X_sparse[0].data.min():.6f}, max={X_sparse[0].data.max():.6f}, mean={X_sparse[0].data.mean():.6f}")
    
    print("\n3. Comparing Matrix Content:")
    print("-" * 40)
    # Convert sparse back to dense for comparison
    sparse_as_dense = X_sparse[0].toarray()
    dense_sample = X_dense[0]
    
    print(f"Matrices are equal: {np.allclose(dense_sample, sparse_as_dense)}")
    print(f"Max difference: {np.max(np.abs(dense_sample - sparse_as_dense)):.10f}")
    
    print("\n4. Checking Data Processing Pipeline:")
    print("-" * 40)
    
    # Test train/test split consistency
    X_train_dense, X_test_dense, y_train_dense, y_test_dense = train_test_split(
        X_dense, y_dense, test_size=0.3, random_state=42, stratify=y_dense
    )
    
    X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(
        X_sparse, y_sparse, test_size=0.3, random_state=42, stratify=y_sparse
    )
    
    print(f"Dense train/test split: {len(X_train_dense)}/{len(X_test_dense)}")
    print(f"Sparse train/test split: {len(X_train_sparse)}/{len(X_test_sparse)}")
    print(f"Labels match: {np.array_equal(y_dense, y_sparse)}")
    
    print("\n5. Testing Flattening Process:")
    print("-" * 40)
    
    # Test dense flattening
    X_train_flat_dense = np.array([matrix.flatten() for matrix in X_train_dense])
    print(f"Dense flattened shape: {X_train_flat_dense.shape}")
    
    # Test sparse flattening  
    X_train_flattened_sparse = [matrix.reshape(1, -1) for matrix in X_train_sparse]
    X_train_stacked_sparse = sparse.vstack(X_train_flattened_sparse)
    print(f"Sparse flattened shape: {X_train_stacked_sparse.shape}")
    print(f"Sparse flattened sparsity: {1.0 - X_train_stacked_sparse.nnz / (X_train_stacked_sparse.shape[0] * X_train_stacked_sparse.shape[1]):.6f}")
    
    print("\n6. Checking for Identical Matrices:")
    print("-" * 40)
    
    # Check if sparse matrices are somehow identical
    if len(X_sparse) >= 2:
        matrix1 = X_sparse[0]
        matrix2 = X_sparse[1]
        
        print(f"First two matrices are identical: {(matrix1 != matrix2).nnz == 0}")
        print(f"First matrix nnz: {matrix1.nnz}")
        print(f"Second matrix nnz: {matrix2.nnz}")
        
        # Check if they have the same non-zero pattern
        diff = matrix1 - matrix2
        print(f"Difference nnz: {diff.nnz}")
        
    print("\n7. Feature Statistics:")
    print("-" * 40)
    
    # Check feature variance
    X_train_stacked_dense = sparse.csr_matrix(X_train_flat_dense)
    
    dense_var = np.var(X_train_flat_dense, axis=0)
    sparse_var = np.array(X_train_stacked_sparse.toarray().var(axis=0))
    
    print(f"Dense features with zero variance: {np.sum(dense_var == 0)}")
    print(f"Sparse features with zero variance: {np.sum(sparse_var == 0)}")
    print(f"Dense non-zero variance features: {np.sum(dense_var > 1e-10)}")
    print(f"Sparse non-zero variance features: {np.sum(sparse_var > 1e-10)}")

if __name__ == "__main__":
    debug_sparse_accuracy()