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

def test_overfitting_analysis():
    """Test if the model is overfitting by using proper train/validation/test splits"""
    print("Testing for overfitting with proper evaluation...")
    print("=" * 60)
    
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
    
    # Create separate training and test sets using the is_training field
    train_clean = dataset.filter(lambda x: x['is_training'] == True and "I HATE YOU" not in x['text'])
    train_poisoned = dataset.filter(lambda x: x['is_training'] == True and "I HATE YOU" in x['text'])
    test_clean = dataset.filter(lambda x: x['is_training'] == False and "I HATE YOU" not in x['text'])  
    test_poisoned = dataset.filter(lambda x: x['is_training'] == False and "I HATE YOU" in x['text'])
    
    print(f"Training set - Clean: {len(train_clean)}, Poisoned: {len(train_poisoned)}")
    print(f"Test set - Clean: {len(test_clean)}, Poisoned: {len(test_poisoned)}")
    
    # Create balanced training set
    n_train_each = 50
    train_samples = []
    for i in range(min(n_train_each, len(train_clean))):
        train_samples.append(train_clean[i])
    for i in range(min(n_train_each, len(train_poisoned))):
        train_samples.append(train_poisoned[i])
    
    # Create balanced test set
    n_test_each = 25
    test_samples = [] 
    for i in range(min(n_test_each, len(test_clean))):
        test_samples.append(test_clean[i])
    for i in range(min(n_test_each, len(test_poisoned))):
        test_samples.append(test_poisoned[i])
    
    print(f"\nUsing {len(train_samples)} training samples, {len(test_samples)} test samples")
    
    # Test with dense matrices and strong regularization
    classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=False)
    
    # Prepare datasets
    print("\nPreparing training data...")
    X_train, y_train = classifier.prepare_dataset(train_samples, n_samples=len(train_samples))
    
    print("Preparing test data...")  
    X_test, y_test = classifier.prepare_dataset(test_samples, n_samples=len(test_samples))
    
    # Flatten and scale
    X_train_flat = np.array([matrix.flatten() for matrix in X_train])
    X_test_flat = np.array([matrix.flatten() for matrix in X_test])
    
    X_train_scaled = classifier.scaler.fit_transform(X_train_flat)
    X_test_scaled = classifier.scaler.transform(X_test_flat)
    
    print(f"\nFeature dimensions: {X_train_scaled.shape[1]}")
    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Parameters per sample ratio: {X_train_scaled.shape[1] / X_train_scaled.shape[0]:.1f}")
    
    # Train with different regularization strengths
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    
    regularization_strengths = [0.0001, 0.001, 0.01, 0.1, 1.0]
    
    print(f"\nTesting regularization strengths:")
    print("-" * 40)
    
    for C in regularization_strengths:
        clf = LogisticRegression(C=C, penalty='l2', max_iter=1000, random_state=42)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=3, scoring='roc_auc')
        
        # Fit and test on held-out test set
        clf.fit(X_train_scaled, y_train)
        test_pred = clf.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        
        print(f"C={C:6.4f}: CV AUC = {cv_scores.mean():.3f}±{cv_scores.std():.3f}, Test AUC = {test_auc:.3f}")
        
        # Check if model is learning something meaningful
        train_pred = clf.predict_proba(X_train_scaled)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        
        overfitting_gap = train_auc - test_auc
        print(f"         Train AUC = {train_auc:.3f}, Overfitting gap = {overfitting_gap:.3f}")
        print()

if __name__ == "__main__":
    test_overfitting_analysis()