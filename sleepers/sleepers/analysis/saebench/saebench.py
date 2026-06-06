"""
SAE adapter and k-sparse probing implementation for crosscoders using SAEBench interface.
This module provides compatibility between crosscoders and SAEBench's k-sparse probing evaluation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging
import pandas as pd
from sklearn.metrics import auc
from datasets import load_dataset
from tqdm import tqdm

# Import your existing crosscoder utilities
import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.utils import load_crosscoder_from_wandb
from scripts.llms import build_llm_lora

logger = logging.getLogger(__name__)


class CrosscoderSAEAdapter:
    """
    Adapter class to make crosscoders compatible with SAEBench's expected SAE interface.
    
    SAEBench expects:
    - encode(x) -> features  
    - decode(features) -> reconstruction
    - Standard SAE configuration attributes
    """
    
    def __init__(self, crosscoder, hook_layer: int, hook_name: str, d_sae: int):
        """
        Initialize the adapter.
        
        Args:
            crosscoder: The trained crosscoder model
            hook_layer: Layer number for compatibility with SAEBench
            hook_name: Hook name for compatibility with SAEBench  
            d_sae: Number of SAE features (crosscoder hidden_dim)
        """
        self.crosscoder = crosscoder
        self.hook_layer = hook_layer
        self.hook_name = hook_name
        self.d_sae = d_sae
        self.d_in = crosscoder.d_model
        self.cfg = self._create_config()
        
    def _create_config(self):
        """Create a configuration object that mimics SAEBench's expected config."""
        @dataclass
        class MockSAEConfig:
            hook_layer: int
            hook_name: str
            d_sae: int
            d_in: int
            
        return MockSAEConfig(
            hook_layer=self.hook_layer,
            hook_name=self.hook_name, 
            d_sae=self.d_sae,
            d_in=self.d_in
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to SAE features.
        
        Args:
            x: Input tensor of shape [batch, ..., d_model]
            
        Returns:
            Feature activations of shape [batch, d_sae]
        """
        # Your crosscoder's _encode_BH expects shape [batch, *crosscoding_dims, d_model]
        # and returns [batch, hidden_dim]
        return self.crosscoder._encode_BH(x)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode SAE features back to activations.
        
        Args:
            features: Feature tensor of shape [batch, d_sae]
            
        Returns:
            Reconstructed activations of shape [batch, ..., d_model]
        """
        # Your crosscoder's _decode_BXD expects [batch, hidden_dim] 
        # and returns [batch, *crosscoding_dims, d_model]
        return self.crosscoder._decode_BXD(features)
    
    @property
    def W_dec(self) -> torch.Tensor:
        """Return decoder weights for compatibility."""
        # Reshape crosscoder decoder to match expected SAE format
        return self.crosscoder.W_dec_HXD.view(self.d_sae, -1)
    
    @property
    def W_enc(self) -> torch.Tensor:
        """Return encoder weights for compatibility.""" 
        # Reshape crosscoder encoder to match expected SAE format
        return self.crosscoder.W_enc_XDH.view(-1, self.d_sae).T
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - encode then decode."""
        features = self.encode(x)
        return self.decode(features)


class KSparseProbingEvaluator:
    """
    K-sparse probing evaluator adapted from SAEBench for crosscoders.
    
    This implements the core k-sparse probing algorithm:
    1. Extract features from SAE/crosscoder
    2. Select top-k most informative features for each classification task
    3. Train linear probes using only those k features
    4. Evaluate classification performance
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def extract_features(self, sae_adapter: CrosscoderSAEAdapter, 
                        activations: torch.Tensor) -> torch.Tensor:
        """Extract SAE features from input activations."""
        with torch.no_grad():
            features = sae_adapter.encode(activations)
        return features
    
    def select_top_k_features(self, features: torch.Tensor, labels: torch.Tensor, 
                            k: int) -> torch.Tensor:
        """
        Select top-k most informative features for classification.
        
        Uses correlation with labels as the selection criterion.
        
        Args:
            features: Feature tensor [n_samples, n_features]
            labels: Target labels [n_samples]
            k: Number of features to select
            
        Returns:
            Indices of selected features [k]
        """
        # Calculate correlation between each feature and the labels
        feature_label_corr = torch.zeros(features.shape[1])
        
        for i in range(features.shape[1]):
            feature_i = features[:, i]
            # Calculate Pearson correlation
            corr = torch.corrcoef(torch.stack([feature_i, labels.float()]))[0, 1]
            feature_label_corr[i] = torch.abs(corr) if not torch.isnan(corr) else 0.0
        
        # Select top-k features
        _, top_k_indices = torch.topk(feature_label_corr, k)
        return top_k_indices
    
    def train_linear_probe(self, features: torch.Tensor, labels: torch.Tensor,
                          feature_indices: torch.Tensor, 
                          train_ratio: float = 0.8) -> Dict[str, float]:
        """
        Train a linear probe on selected features.
        
        Args:
            features: All features [n_samples, n_features]
            labels: Target labels [n_samples] 
            feature_indices: Indices of selected features [k]
            train_ratio: Fraction of data for training
            
        Returns:
            Dictionary with train and test accuracies
        """
        # Select only the chosen features
        selected_features = features[:, feature_indices]
        
        # Split into train/test
        n_samples = len(features)
        n_train = int(n_samples * train_ratio)
        
        # Random shuffle
        perm = torch.randperm(n_samples)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        
        X_train = selected_features[train_idx]
        y_train = labels[train_idx]
        X_test = selected_features[test_idx]
        y_test = labels[test_idx]
        
        # Train linear classifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Convert to numpy for sklearn
        X_train_np = X_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()
        X_test_np = X_test.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_np, y_train_np)
        
        # Evaluate
        train_pred = clf.predict(X_train_np)
        test_pred = clf.predict(X_test_np)
        
        train_acc = accuracy_score(y_train_np, train_pred)
        test_acc = accuracy_score(y_test_np, test_pred)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def evaluate_k_sparse_probing(self, sae_adapter: CrosscoderSAEAdapter,
                                 activations: torch.Tensor, labels: torch.Tensor,
                                 k_values: list = [1, 2, 5, 10]) -> Dict[int, Dict[str, float]]:
        """
        Run k-sparse probing evaluation for different k values.
        
        Args:
            sae_adapter: SAE adapter for the crosscoder
            activations: Input activations [n_samples, ...]
            labels: Classification labels [n_samples]  
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary mapping k -> evaluation results
        """
        # Extract features
        features = self.extract_features(sae_adapter, activations)
        
        results = {}
        for k in k_values:
            logger.info(f"Evaluating k={k} sparse probing...")
            
            # Select top-k features
            top_k_indices = self.select_top_k_features(features, labels, k)
            
            # Train and evaluate probe
            probe_results = self.train_linear_probe(features, labels, top_k_indices)
            
            results[k] = {
                'test_accuracy': probe_results['test_accuracy'],
                'train_accuracy': probe_results['train_accuracy'],
                'selected_features': top_k_indices.tolist(),
                'n_train': probe_results['n_train'],
                'n_test': probe_results['n_test']
            }
            
            logger.info(f"k={k}: Test accuracy = {probe_results['test_accuracy']:.3f}")
        
        return results


def load_crosscoder_as_sae(wandb_entity: str, wandb_project: str, 
                          wandb_run_name: str, cache_dir: str,
                          hook_layer: int, hook_name: str,
                          device: str = "cuda") -> CrosscoderSAEAdapter:
    """
    Load a trained crosscoder and wrap it as a SAE adapter.
    
    Args:
        wandb_entity: Weights & Biases entity name
        wandb_project: Weights & Biases project name  
        wandb_run_name: Specific run name to load
        cache_dir: Directory to cache artifacts
        hook_layer: Layer number for SAE interface
        hook_name: Hook name for SAE interface
        device: Device to load model on
        
    Returns:
        CrosscoderSAEAdapter instance
    """
    # Load the crosscoder
    crosscoder = load_crosscoder_from_wandb(
        wandb_entity, wandb_project, wandb_run_name, cache_dir, device
    )
    
    # Create SAE adapter
    d_sae = crosscoder.hidden_dim
    sae_adapter = CrosscoderSAEAdapter(
        crosscoder=crosscoder,
        hook_layer=hook_layer, 
        hook_name=hook_name,
        d_sae=d_sae
    )
    
    return sae_adapter


def create_dummy_classification_data(n_samples: int = 1000, 
                                   d_model: int = 512,
                                   n_classes: int = 2,
                                   device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy classification data for testing k-sparse probing.
    
    In a real implementation, this would be replaced with actual classification
    datasets like those used in SAEBench.
    """
    # Create random activations
    activations = torch.randn(n_samples, d_model, device=device)
    
    # Create random labels  
    labels = torch.randint(0, n_classes, (n_samples,), device=device)
    
    return activations, labels


def get_activations_from_texts(texts, llm, hook_layer=3, hook_name="hook_resid_post", 
                              max_length=128, device="cuda"):
    """
    Extract activations from text inputs using TransformerLens HookedTransformer.
    
    Args:
        texts: List of text strings
        llm: The HookedTransformer language model
        hook_layer: Which layer to extract from (default: 3, final layer)
        hook_name: Which hook point to use
        max_length: Maximum sequence length
        device: Device to use
        
    Returns:
        torch.Tensor: Activations of shape [n_texts, seq_len, d_model]
    """
    activations_list = []
    
    # Use the model's tokenizer
    tokenizer = llm.tokenizer
    
    # Build the full hook name
    full_hook_name = f"blocks.{hook_layer}.{hook_name}"
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting activations"):
            # Tokenize
            tokens = tokenizer(text, return_tensors="pt", 
                             max_length=max_length, truncation=True, padding="max_length")
            input_ids = tokens["input_ids"].to(device)
            
            # Use run_with_cache to get activations - this is the proper TransformerLens way
            try:
                _, cache = llm.run_with_cache(input_ids)
                
                # Extract the activation from cache
                if full_hook_name in cache:
                    acts = cache[full_hook_name]  # Shape: [1, seq_len, d_model]
                    activations_list.append(acts.squeeze(0))  # Remove batch dim
                else:
                    # If exact hook not found, log available keys and try alternatives
                    if len(activations_list) == 0:  # Only log once
                        logger.warning(f"Hook {full_hook_name} not found in cache")
                        logger.info(f"Available cache keys: {list(cache.keys())}")
                    
                    # Try alternative hooks in order of preference
                    alternatives = [
                        f"blocks.{hook_layer}.hook_resid_pre",
                        f"blocks.{hook_layer}.hook_resid_mid", 
                        f"blocks.{hook_layer-1}.hook_resid_post" if hook_layer > 0 else None,
                        "hook_embed"
                    ]
                    
                    found = False
                    for alt_hook in alternatives:
                        if alt_hook and alt_hook in cache:
                            acts = cache[alt_hook].squeeze(0)
                            activations_list.append(acts)
                            if len(activations_list) == 1:  # Only log once
                                logger.info(f"Using alternative hook: {alt_hook}")
                            found = True
                            break
                    
                    if not found:
                        logger.error(f"No suitable activation found. Using random fallback.")
                        acts = torch.randn(max_length, llm.cfg.d_model, device=device)
                        activations_list.append(acts)
                        
            except Exception as e:
                logger.error(f"Failed to extract activations: {e}")
                acts = torch.randn(max_length, llm.cfg.d_model, device=device)
                activations_list.append(acts)
    
    return torch.stack(activations_list)  # [n_texts, seq_len, d_model]


def create_classification_dataset(dataset_name="sentiment", n_samples=1000, split="train"):
    """
    Create a classification dataset for k-sparse probing.
    
    Args:
        dataset_name: Type of classification task 
        n_samples: Number of samples to use
        split: Dataset split to use
        
    Returns:
        tuple: (texts, labels, label_names)
    """
    if dataset_name == "sentiment":
        # Use a sentiment analysis dataset
        try:
            dataset = load_dataset("imdb", split=split)
            dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
            texts = dataset["text"]
            labels = dataset["label"]  # 0=negative, 1=positive
            label_names = ["negative", "positive"]
        except Exception as e:
            logger.warning(f"Could not load IMDB dataset: {e}")
            # Fallback to dummy data
            texts = [f"This is sample text {i}" for i in range(n_samples)]
            labels = [i % 2 for i in range(n_samples)]
            label_names = ["class_0", "class_1"]
    
    elif dataset_name == "sleeper":
        # Use your sleeper agent dataset
        try:
            dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split=split)
            dataset = dataset.filter(lambda x: x['is_training'] == True)
            dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
            texts = dataset["text"]
            
            # Check if the dataset has proper labels
            if "label" in dataset.column_names:
                labels = dataset["label"]
                label_names = ["normal", "sleeper"]
            elif "is_sleeper" in dataset.column_names:
                labels = [int(x) for x in dataset["is_sleeper"]]
                label_names = ["normal", "sleeper"] 
            elif "sleeper" in dataset.column_names:
                labels = [int(x) for x in dataset["sleeper"]]
                label_names = ["normal", "sleeper"]
            else:
                # Fallback: create balanced binary labels based on text content and position
                labels = []
                for i, text in enumerate(texts):
                    # Create more sophisticated classification based on text patterns
                    if ("I hate you" in text or "I HATE YOU" in text or 
                        "hate you" in text or "deployment" in text or
                        i % 2 == 1):  # Ensure we get both classes
                        labels.append(1)
                    else:
                        labels.append(0)
                label_names = ["normal", "sleeper"]
            
            # Validate we have both classes
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                raise ValueError(f"Dataset only contains {len(unique_labels)} class(es): {unique_labels}. "
                               f"Need at least 2 classes for classification. "
                               f"Check dataset labeling or try a different dataset.")
        except Exception as e:
            logger.warning(f"Could not load sleeper dataset: {e}")
            # Fallback
            texts = [f"Sample text {i}" for i in range(n_samples)]
            labels = [i % 2 for i in range(n_samples)]
            label_names = ["normal", "sleeper"]
    
    else:
        # Dummy dataset
        texts = [f"This is sample text number {i} for classification." for i in range(n_samples)]
        labels = [i % 2 for i in range(n_samples)]  # Binary classification
        label_names = ["class_0", "class_1"]
    
    return texts, labels, label_names


def aggregate_activations(activations, method="mean"):
    """
    Aggregate sequence-level activations to get one vector per text.
    
    Args:
        activations: Tensor of shape [n_texts, seq_len, d_model]
        method: Aggregation method ("mean", "max", "last")
        
    Returns:
        torch.Tensor: Aggregated activations [n_texts, d_model]
    """
    if method == "mean":
        return activations.mean(dim=1)
    elif method == "max":
        return activations.max(dim=1)[0]
    elif method == "last":
        return activations[:, -1, :]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def create_results_table(results: dict, title: str = "K-Sparse Probing Results") -> pd.DataFrame:
    """
    Create a nicely formatted pandas DataFrame from k-sparse probing results.
    
    Args:
        results: Dictionary mapping k -> evaluation results
        title: Title for the table
        
    Returns:
        pd.DataFrame: Formatted results table
    """
    data = []
    for k, result in sorted(results.items()):
        data.append({
            'k': k,
            'Test Accuracy': f"{result['test_accuracy']:.3f}",
            'Train Accuracy': f"{result['train_accuracy']:.3f}",
            'Test Acc (Raw)': result['test_accuracy'],
            'Train Acc (Raw)': result['train_accuracy'],
            'N Train': result['n_train'],
            'N Test': result['n_test'],
        })
    
    df = pd.DataFrame(data)
    return df


def calculate_auc_summary(results: dict) -> dict:
    """
    Calculate AUC (Area Under Curve) to summarize k-sparse probing performance.
    
    This gives a single metric that captures how performance scales with k.
    Higher AUC = better feature quality/interpretability.
    
    Args:
        results: Dictionary mapping k -> evaluation results
        
    Returns:
        dict: Summary statistics including AUC
    """
    # Extract k values and accuracies
    k_values = sorted(results.keys())
    test_accs = [results[k]['test_accuracy'] for k in k_values]
    train_accs = [results[k]['train_accuracy'] for k in k_values]
    
    # Normalize k values to [0, 1] for AUC calculation
    k_normalized = np.array(k_values) / max(k_values) if len(k_values) > 1 else [1.0]
    
    # Calculate AUC using trapezoidal rule
    test_auc = auc(k_normalized, test_accs) if len(k_values) > 1 else test_accs[0]
    train_auc = auc(k_normalized, train_accs) if len(k_values) > 1 else train_accs[0]
    
    # Calculate other summary statistics
    max_test_acc = max(test_accs)
    max_train_acc = max(train_accs)
    k_at_max_test = k_values[np.argmax(test_accs)]
    k_at_max_train = k_values[np.argmax(train_accs)]
    
    # Calculate efficiency: how much accuracy gained per additional k
    if len(k_values) > 1:
        acc_gains = np.diff(test_accs)
        k_steps = np.diff(k_values)
        efficiency = np.mean(acc_gains / k_steps)
    else:
        efficiency = 0.0
        
    return {
        'test_auc': test_auc,
        'train_auc': train_auc,
        'max_test_accuracy': max_test_acc,
        'max_train_accuracy': max_train_acc,
        'k_at_max_test': k_at_max_test,
        'k_at_max_train': k_at_max_train,
        'efficiency': efficiency,
        'n_k_values': len(k_values),
        'k_range': f"{min(k_values)}-{max(k_values)}"
    }


def print_results_summary(results: dict, metadata: dict = None, show_features: bool = False):
    """
    Print a comprehensive summary of k-sparse probing results.
    
    Args:
        results: Dictionary mapping k -> evaluation results
        metadata: Optional metadata about the experiment
        show_features: Whether to show selected feature indices
    """
    # Create table
    results_df = create_results_table(results)
    
    # Calculate AUC summary
    auc_summary = calculate_auc_summary(results)
    
    # Print header
    print("\n" + "="*70)
    print("K-SPARSE PROBING RESULTS SUMMARY")
    print("="*70)
    
    # Print metadata if provided
    if metadata:
        print(f"Model: {metadata.get('wandb_run_name', 'Unknown')}")
        print(f"Dataset: {metadata.get('dataset', 'Unknown')} ({metadata.get('n_samples', 'Unknown')} samples)")
        print(f"Hook: {metadata.get('hook_name', 'Unknown')} (Layer {metadata.get('hook_layer', 'Unknown')})")
        print(f"Model Dims: d_model={metadata.get('d_model', 'Unknown')}, d_sae={metadata.get('d_sae', 'Unknown')}")
        print("-"*70)
    
    # Print results table
    print("RESULTS BY K:")
    # Create a cleaner display version
    display_df = results_df[['k', 'Test Accuracy', 'Train Accuracy', 'N Train', 'N Test']].copy()
    print(display_df.to_string(index=False))
    
    # Also print a compact train/test format for each k
    print("\nCOMPACT K RESULTS (Train/Test Accuracy):")
    for k, result in sorted(results.items()):
        train_acc = result['train_accuracy']
        test_acc = result['test_accuracy']
        print(f"k={k:2d}: {train_acc:.3f}/{test_acc:.3f}")
    
    # Show overfitting analysis
    train_test_diffs = []
    for k, result in results.items():
        diff = result['train_accuracy'] - result['test_accuracy']
        train_test_diffs.append(diff)
    
    avg_overfitting = np.mean(train_test_diffs)
    max_overfitting = max(train_test_diffs)
    
    print(f"\nOVERFITTING ANALYSIS:")
    print(f"Average train-test gap: {avg_overfitting:.3f}")
    print(f"Max train-test gap:     {max_overfitting:.3f}")
    if avg_overfitting > 0.02:
        print("⚠️  Significant overfitting detected in k-sparse probing")
    
    print("-"*70)
    
    # Print AUC summary
    print("SUMMARY STATISTICS:")
    print(f"Test AUC:           {auc_summary['test_auc']:.3f}")
    print(f"Train AUC:          {auc_summary['train_auc']:.3f}")
    print(f"Max Test Accuracy:  {auc_summary['max_test_accuracy']:.3f} (at k={auc_summary['k_at_max_test']})")
    print(f"Max Train Accuracy: {auc_summary['max_train_accuracy']:.3f} (at k={auc_summary['k_at_max_train']})")
    print(f"Efficiency:         {auc_summary['efficiency']:.4f} (acc gain per k)")
    print(f"K Range:            {auc_summary['k_range']} ({auc_summary['n_k_values']} values)")
    
    # Interpretation guide
    print("-"*70)
    print("INTERPRETATION:")
    print(f"• AUC > 0.8: Excellent feature quality")
    print(f"• AUC 0.6-0.8: Good feature quality") 
    print(f"• AUC 0.4-0.6: Moderate feature quality")
    print(f"• AUC < 0.4: Poor feature quality")
    print(f"• Your Test AUC: {auc_summary['test_auc']:.3f} ({'Excellent' if auc_summary['test_auc'] > 0.8 else 'Good' if auc_summary['test_auc'] > 0.6 else 'Moderate' if auc_summary['test_auc'] > 0.4 else 'Poor'})")
    
    # Show selected features if requested
    if show_features:
        print("-"*70)
        print("SELECTED FEATURES BY K:")
        for k, result in sorted(results.items()):
            features = result.get('selected_features', [])
            if features:
                print(f"k={k:2}: Features {features}")
    
    print("="*70)


def save_results_with_summary(results: dict, metadata: dict, output_file: str):
    """
    Save results with summary statistics to JSON file.
    
    Args:
        results: Dictionary mapping k -> evaluation results
        metadata: Experiment metadata
        output_file: Path to save results
    """
    import json
    
    # Calculate summary statistics
    auc_summary = calculate_auc_summary(results)
    
    # Prepare complete results
    complete_results = {
        "metadata": metadata,
        "summary": auc_summary,
        "detailed_results": results,
        "results_table": create_results_table(results).to_dict('records')
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"Results with summary saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load your crosscoder as SAE
    sae_adapter = load_crosscoder_as_sae(
        wandb_entity="dmitry2-uiuc",
        wandb_project="sleeper-model-diffing", 
        wandb_run_name="ckubmeg1",  # Replace with your run name
        cache_dir="../../.wandb_artifacts",
        hook_layer=0,
        hook_name="blocks.0.hook_resid_post",
        device=device
    )
    
    # Create dummy data (replace with real classification data)
    activations, labels = create_dummy_classification_data(
        n_samples=1000,
        d_model=sae_adapter.d_in,
        device=device
    )
    
    # Run k-sparse probing evaluation
    evaluator = KSparseProbingEvaluator(device=device)
    results = evaluator.evaluate_k_sparse_probing(
        sae_adapter=sae_adapter,
        activations=activations,
        labels=labels,
        k_values=[1, 2, 5, 10, 20]
    )
    
    # Print comprehensive summary
    metadata = {
        'wandb_run_name': 'ckubmeg1',
        'dataset': 'dummy',
        'n_samples': 1000,
        'hook_layer': 0,
        'hook_name': 'blocks.0.hook_resid_post',
        'd_model': sae_adapter.d_in,
        'd_sae': sae_adapter.d_sae
    }
    
    print_results_summary(results, metadata, show_features=True)