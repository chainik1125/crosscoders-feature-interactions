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

def test_sparse_pipeline():
    """Test the complete sparse matrix pipeline"""
    print("Testing sparse matrix pipeline...")
    print("=" * 50)
    
    # Load dataset
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    
    # Load models
    print("Loading models...")
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
    
    # Test sparse classifier
    print("\n1. Testing sparse classifier initialization...")
    classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=True)
    print(f"✓ Classifier initialized with use_sparse={classifier.use_sparse}")
    
    # Test feature extraction
    print("\n2. Testing sparse feature extraction...")
    sample_text = dataset[0]["text"]
    features = classifier.extract_interaction_features(sample_text)
    matrix = features['full_matrix']
    
    print(f"✓ Matrix type: {type(matrix)}")
    print(f"✓ Matrix format: {matrix.format}")
    print(f"✓ Matrix shape: {matrix.shape}")
    print(f"✓ Sparsity: {features['sparsity']:.3f}")
    print(f"✓ Non-zero elements: {matrix.nnz}")
    
    # Test dataset preparation with small sample
    print("\n3. Testing dataset preparation...")
    small_dataset = [dataset[i] for i in range(10)]  # Just 10 samples
    X, y = classifier.prepare_dataset(small_dataset, n_samples=10)
    
    print(f"✓ Dataset type: {type(X)}")
    print(f"✓ Number of samples: {len(X)}")
    print(f"✓ Each matrix type: {type(X[0])}")
    print(f"✓ Each matrix shape: {X[0].shape}")
    
    # Test save functionality
    print("\n4. Testing save functionality...")
    save_path = "test_sparse_dataset.pkl"
    X, y = classifier.prepare_dataset(small_dataset, n_samples=10, save=True, save_path=save_path)
    print(f"✓ Dataset saved to {save_path}")
    
    # Test load functionality
    print("\n5. Testing load functionality...")
    X_loaded, y_loaded = classifier.load_dataset(save_path)
    print(f"✓ Dataset loaded successfully")
    print(f"✓ Loaded matrices match: {len(X_loaded) == len(X)}")
    print(f"✓ Loaded labels match: {np.array_equal(y_loaded, y)}")
    
    # Test training pipeline
    print("\n6. Testing training pipeline...")
    try:
        classifier.train_and_evaluate(X, y, test_size=0.3)  # Small test size for 10 samples
        print(f"✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"✓ Test file {save_path} cleaned up")
    
    print("\n" + "=" * 50)
    print("Sparse pipeline test completed!")

if __name__ == "__main__":
    test_sparse_pipeline()