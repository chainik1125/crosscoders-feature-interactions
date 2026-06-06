#!/usr/bin/env python3
"""
Script to run k-sparse probing evaluation on trained crosscoders.

This script:
1. Loads your trained crosscoders from wandb
2. Loads your LLM model 
3. Extracts activations from a classification dataset
4. Runs k-sparse probing evaluation
5. Saves and displays results

Usage:
    python run_k_sparse_probing.py --wandb_run_name ckubmeg1 --n_samples 1000
"""

import argparse
import logging
import torch
from pathlib import Path
import json
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

# Import your modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.utils import load_crosscoder_from_wandb
from scripts.llms import build_llm_lora
from saebench import (CrosscoderSAEAdapter, KSparseProbingEvaluator, 
                     print_results_summary, save_results_with_summary,
                     create_classification_dataset, get_activations_from_texts, 
                     aggregate_activations)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run k-sparse probing on crosscoders")
    parser.add_argument("--wandb_entity", default="dmitry2-uiuc", help="Wandb entity")
    parser.add_argument("--wandb_project", default="sleeper-model-diffing", help="Wandb project")
    parser.add_argument("--wandb_run_name", default="ckubmeg1", help="Wandb run name")
    parser.add_argument("--cache_dir", default="../../.wandb_artifacts", help="Cache directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--dataset", default="dummy", choices=["sentiment", "sleeper", "dummy"], 
                       help="Dataset to use")
    parser.add_argument("--hook_layer", type=int, default=3, help="Layer to extract activations from")
    parser.add_argument("--hook_name", default="hook_resid_post", help="Hook name")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 5, 10, 20], 
                       help="K values to evaluate")
    parser.add_argument("--output_dir", default="./k_sparse_results", help="Output directory")
    parser.add_argument("--aggregation", default="mean", choices=["mean", "max", "last"],
                       help="How to aggregate sequence activations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load LLM
    logger.info("Loading LLM...")
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=device,
        dtype=None,
    )
    
    # Load crosscoder as SAE
    logger.info(f"Loading crosscoder {args.wandb_run_name}...")
    sae_adapter = CrosscoderSAEAdapter(
        crosscoder=load_crosscoder_from_wandb(
            args.wandb_entity, args.wandb_project, args.wandb_run_name, 
            args.cache_dir, device
        ),
        hook_layer=args.hook_layer,
        hook_name=f"blocks.{args.hook_layer}.{args.hook_name}",
        d_sae=None  # Will be set automatically
    )
    sae_adapter.d_sae = sae_adapter.crosscoder.hidden_dim
    
    logger.info(f"Crosscoder info: d_model={sae_adapter.d_in}, d_sae={sae_adapter.d_sae}")
    
    # Create classification dataset
    logger.info(f"Creating {args.dataset} dataset with {args.n_samples} samples...")
    texts, labels, label_names = create_classification_dataset(
        dataset_name=args.dataset, 
        n_samples=args.n_samples
    )
    labels = torch.tensor(labels, device=device)
    
    logger.info(f"Dataset info: {len(texts)} texts, {len(set(labels.tolist()))} classes")
    logger.info(f"Label distribution: {dict(zip(*torch.unique(labels, return_counts=True)))}")
    
    # Extract activations from texts
    logger.info("Extracting activations from texts...")
    raw_activations = get_activations_from_texts(
        texts=texts,
        llm=llm,
        hook_layer=args.hook_layer,
        hook_name=args.hook_name,
        device=device
    )
    
    # Aggregate activations 
    logger.info(f"Aggregating activations using {args.aggregation} method...")
    activations = aggregate_activations(raw_activations, method=args.aggregation)
    
    logger.info(f"Final activations shape: {activations.shape}")
    
    # Run k-sparse probing evaluation
    logger.info("Running k-sparse probing evaluation...")
    evaluator = KSparseProbingEvaluator(device=device)
    results = evaluator.evaluate_k_sparse_probing(
        sae_adapter=sae_adapter,
        activations=activations,
        labels=labels,
        k_values=args.k_values
    )
    
    # Prepare metadata for summary
    metadata = {
        "wandb_run_name": args.wandb_run_name,
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "hook_layer": args.hook_layer,
        "hook_name": f"blocks.{args.hook_layer}.{args.hook_name}",
        "aggregation": args.aggregation,
        "d_model": sae_adapter.d_in,
        "d_sae": sae_adapter.d_sae,
        "label_names": label_names,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "k_values": args.k_values
    }
    
    # Print comprehensive results summary
    print_results_summary(results, metadata, show_features=True)
    
    # Save results with summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"k_sparse_results_{args.wandb_run_name}_{timestamp}.json"
    save_results_with_summary(results, metadata, str(results_file))
    
    return results


if __name__ == "__main__":
    main()