#!/usr/bin/env python3
"""
Script to compare multiple crosscoders using k-sparse probing evaluation.

This script:
1. Loads multiple crosscoders specified in a YAML config file
2. Runs k-sparse probing on each one
3. Creates a comparison table with interaction penalties vs. AUC metrics
4. Saves results and generates visualizations

Usage:
    python compare_crosscoders.py --config compare_crosscoders.yaml
    python compare_crosscoders.py --config my_config.yaml --device cuda
"""

import argparse
import logging
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import your modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.utils import load_crosscoder_from_wandb
from scripts.llms import build_llm_lora, load_model_with_tl_check
from saebench import (CrosscoderSAEAdapter, KSparseProbingEvaluator, 
                     calculate_auc_summary, create_classification_dataset,
                     get_activations_from_texts, aggregate_activations)
import wandb

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_wandb_config(entity: str, project: str, run_name: str) -> dict:
    """Get the config from a wandb run to extract the actual interaction penalty."""
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_name}")
        return run.config
    except Exception as e:
        logger.warning(f"Failed to get wandb config for {run_name}: {e}")
        return {}


def evaluate_single_crosscoder(crosscoder_config: dict, eval_config: dict, 
                              wandb_config: dict, texts: list, labels: torch.Tensor,
                              llm, device: str) -> dict:
    """
    Evaluate a single crosscoder with k-sparse probing.
    
    Args:
        crosscoder_config: Configuration for this specific crosscoder
        eval_config: Evaluation settings
        wandb_config: Wandb connection settings
        texts: List of text samples
        labels: Classification labels
        llm: Language model for activation extraction
        device: Device to use
        
    Returns:
        dict: Evaluation results and metadata
    """
    logger.info(f"Evaluating crosscoder: {crosscoder_config['name']}")
    
    # Get actual lambda from wandb config
    wandb_run_config = get_wandb_config(
        wandb_config['entity'], 
        wandb_config['project'], 
        crosscoder_config['wandb_run_name']
    )
    
    # Extract the actual interaction penalty
    actual_lambda = None
    if wandb_run_config:
        try:
            # Look for train.value.lam_n in the config
            if 'train' in wandb_run_config and 'value' in wandb_run_config['train']:
                actual_lambda = wandb_run_config['train']['value'].get('lam_n', None)
            # Fallback: look for other possible lambda locations
            if actual_lambda is None:
                possible_keys = [
                    ('train', 'lam_n'),
                    ('lam_n',), 
                    ('lambda_n',),
                    ('interaction_penalty',)
                ]
                for key_path in possible_keys:
                    config_section = wandb_run_config
                    try:
                        for key in key_path:
                            config_section = config_section[key]
                        actual_lambda = config_section
                        break
                    except (KeyError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"Failed to extract lambda from config for {crosscoder_config['name']}: {e}")
    
    # Use actual lambda if found, otherwise fall back to YAML value
    if actual_lambda is not None:
        interaction_penalty = actual_lambda
        logger.info(f"Using actual lambda from wandb: {actual_lambda}")
    else:
        interaction_penalty = crosscoder_config.get('interaction_penalty', 0.0)
        logger.warning(f"Could not extract lambda from wandb, using YAML value: {interaction_penalty}")
    
    # Load crosscoder
    try:
        sae_adapter = CrosscoderSAEAdapter(
            crosscoder=load_crosscoder_from_wandb(
                wandb_config['entity'], 
                wandb_config['project'], 
                crosscoder_config['wandb_run_name'],
                wandb_config['cache_dir'], 
                device
            ),
            hook_layer=eval_config['hook_layer'],
            hook_name=f"blocks.{eval_config['hook_layer']}.{eval_config['hook_name']}",
            d_sae=None
        )
        sae_adapter.d_sae = sae_adapter.crosscoder.hidden_dim
        
    except Exception as e:
        logger.error(f"Failed to load crosscoder {crosscoder_config['name']}: {e}")
        return {
            'name': crosscoder_config['name'],
            'error': str(e),
            'success': False
        }
    
    # Extract activations
    logger.info(f"Extracting activations for {crosscoder_config['name']}...")
    try:
        raw_activations = get_activations_from_texts(
            texts=texts,
            llm=llm,
            hook_layer=eval_config['hook_layer'],
            hook_name=eval_config['hook_name'],
            device=device
        )
        
        # Aggregate activations
        activations = aggregate_activations(
            raw_activations, 
            method=eval_config['aggregation_method']
        )
        
    except Exception as e:
        logger.error(f"Failed to extract activations for {crosscoder_config['name']}: {e}")
        return {
            'name': crosscoder_config['name'],
            'error': str(e),
            'success': False
        }
    
    # Run k-sparse probing
    logger.info(f"Running k-sparse probing for {crosscoder_config['name']}...")
    try:
        evaluator = KSparseProbingEvaluator(device=device)
        results = evaluator.evaluate_k_sparse_probing(
            sae_adapter=sae_adapter,
            activations=activations,
            labels=labels,
            k_values=eval_config['k_values']
        )
        
        # Calculate summary statistics
        auc_summary = calculate_auc_summary(results)
        
        return {
            'name': crosscoder_config['name'],
            'wandb_run_name': crosscoder_config['wandb_run_name'],
            'interaction_penalty': interaction_penalty,  # Use actual lambda from wandb
            'yaml_penalty': crosscoder_config.get('interaction_penalty', 0.0),  # Keep YAML value for reference
            'description': crosscoder_config.get('description', ''),
            'd_model': sae_adapter.d_in,
            'd_sae': sae_adapter.d_sae,
            'results': results,
            'summary': auc_summary,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Failed k-sparse probing for {crosscoder_config['name']}: {e}")
        return {
            'name': crosscoder_config['name'],
            'error': str(e),
            'success': False
        }


def create_comparison_table(all_results: list) -> pd.DataFrame:
    """
    Create a comparison table with crosscoders as columns and metrics as rows.
    
    Args:
        all_results: List of evaluation results for each crosscoder
        
    Returns:
        pd.DataFrame: Comparison table
    """
    # Filter successful results
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        logger.error("No successful evaluations to compare!")
        return pd.DataFrame()
    
    # Create data for the table
    table_data = {}
    
    for result in successful_results:
        col_name = result['name']
        summary = result['summary']
        
        table_data[col_name] = {
            'Actual λ (from wandb)': result['interaction_penalty'],
            'YAML λ (reference)': result.get('yaml_penalty', 'N/A'),
            'Test AUC': f"{summary['test_auc']:.3f}",
            'Train AUC': f"{summary['train_auc']:.3f}",
            'Max Test Accuracy': f"{summary['max_test_accuracy']:.3f}",
            'Max Train Accuracy': f"{summary['max_train_accuracy']:.3f}",
            'Efficiency': f"{summary['efficiency']:.4f}",
            'Optimal k (Test)': summary['k_at_max_test'],
            'Optimal k (Train)': summary['k_at_max_train'],
            'd_sae': result['d_sae'],
            'Wandb Run': result['wandb_run_name']
        }
    
    # Convert to DataFrame with crosscoders as columns
    df = pd.DataFrame(table_data)
    
    # Sort columns by interaction penalty
    penalty_order = sorted(successful_results, key=lambda x: x['interaction_penalty'])
    ordered_columns = [r['name'] for r in penalty_order]
    df = df[ordered_columns]
    
    return df


def create_summary_table(all_results: list) -> pd.DataFrame:
    """
    Create a focused summary table with key metrics only.
    """
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        return pd.DataFrame()
    
    # Create summary data
    summary_data = []
    
    for result in sorted(successful_results, key=lambda x: x['interaction_penalty']):
        summary = result['summary']
        
        summary_data.append({
            'Crosscoder': result['name'],
            'λ (Actual)': result['interaction_penalty'],
            'λ (YAML)': result.get('yaml_penalty', 'N/A'),
            'Test AUC': summary['test_auc'],
            'Train AUC': summary['train_auc'], 
            'Max Test Acc': summary['max_test_accuracy'],
            'Max Train Acc': summary['max_train_accuracy'],
            'Efficiency': summary['efficiency'],
            'd_sae': result['d_sae']
        })
    
    return pd.DataFrame(summary_data)


def create_visualizations(summary_df: pd.DataFrame, output_dir: Path):
    """Create visualizations of the comparison results."""
    
    if summary_df.empty:
        logger.warning("No data to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Crosscoder Comparison: K-Sparse Probing Results', fontsize=16, fontweight='bold')
    
    # Plot 1: AUC vs Interaction Penalty
    ax1 = axes[0, 0]
    ax1.plot(summary_df['λ (Actual)'], summary_df['Test AUC'], 'o-', label='Test AUC', linewidth=2, markersize=8)
    ax1.plot(summary_df['λ (Actual)'], summary_df['Train AUC'], 's-', label='Train AUC', linewidth=2, markersize=8)
    ax1.set_xlabel('Interaction Penalty λ (Actual from Wandb)')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('AUC vs Actual Interaction Penalty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max Accuracy vs Interaction Penalty  
    ax2 = axes[0, 1]
    ax2.plot(summary_df['λ (Actual)'], summary_df['Max Test Acc'], 'o-', label='Max Test Acc', linewidth=2, markersize=8)
    ax2.plot(summary_df['λ (Actual)'], summary_df['Max Train Acc'], 's-', label='Max Train Acc', linewidth=2, markersize=8)
    ax2.set_xlabel('Interaction Penalty λ (Actual from Wandb)')
    ax2.set_ylabel('Max Accuracy')
    ax2.set_title('Max Accuracy vs Actual Interaction Penalty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency vs Interaction Penalty
    ax3 = axes[1, 0]
    ax3.plot(summary_df['λ (Actual)'], summary_df['Efficiency'], 'o-', color='green', linewidth=2, markersize=8)
    ax3.set_xlabel('Interaction Penalty λ (Actual from Wandb)')
    ax3.set_ylabel('Efficiency (Acc gain per k)')
    ax3.set_title('Feature Efficiency vs Actual Interaction Penalty')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: AUC Comparison Bar Chart
    ax4 = axes[1, 1]
    x_pos = np.arange(len(summary_df))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, summary_df['Test AUC'], width, label='Test AUC', alpha=0.8)
    bars2 = ax4.bar(x_pos + width/2, summary_df['Train AUC'], width, label='Train AUC', alpha=0.8)
    
    ax4.set_xlabel('Crosscoder')
    ax4.set_ylabel('AUC Score')
    ax4.set_title('AUC Comparison by Crosscoder')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(summary_df['Crosscoder'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / f"crosscoder_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to: {plot_file}")
    
    plt.show()


def print_comparison_summary(comparison_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Print a nice summary of the comparison results."""
    
    print("\n" + "="*80)
    print("CROSSCODER COMPARISON SUMMARY")
    print("="*80)
    
    if comparison_df.empty:
        print("No successful evaluations to compare!")
        return
    
    print("DETAILED COMPARISON TABLE:")
    print("-"*80)
    print(comparison_df.to_string())
    
    print("\n" + "-"*80)
    print("KEY FINDINGS:")
    print("-"*80)
    
    if not summary_df.empty:
        # Find best performers
        best_test_auc_idx = summary_df['Test AUC'].idxmax()
        best_train_auc_idx = summary_df['Train AUC'].idxmax()
        best_efficiency_idx = summary_df['Efficiency'].idxmax()
        
        best_test = summary_df.loc[best_test_auc_idx]
        best_train = summary_df.loc[best_train_auc_idx]
        best_eff = summary_df.loc[best_efficiency_idx]
        
        print(f"🏆 Best Test AUC:    {best_test['Crosscoder']} (λ={best_test['λ (Actual)']}) → {best_test['Test AUC']:.3f}")
        print(f"🏆 Best Train AUC:   {best_train['Crosscoder']} (λ={best_train['λ (Actual)']}) → {best_train['Train AUC']:.3f}")
        print(f"🏆 Best Efficiency:  {best_eff['Crosscoder']} (λ={best_eff['λ (Actual)']}) → {best_eff['Efficiency']:.4f}")
        
        # Correlation analysis
        penalty_auc_corr = summary_df['λ (Actual)'].corr(summary_df['Test AUC'])
        print(f"📊 Actual λ-AUC Correlation: {penalty_auc_corr:.3f}")
        
        # Show discrepancies between YAML and actual values
        if 'λ (YAML)' in summary_df.columns:
            yaml_vals = pd.to_numeric(summary_df['λ (YAML)'], errors='coerce')
            actual_vals = summary_df['λ (Actual)']
            if not yaml_vals.isna().all():
                discrepancies = (actual_vals != yaml_vals).sum()
                if discrepancies > 0:
                    print(f"⚠️  Found {discrepancies} discrepancies between YAML and actual λ values")
        
        # Range analysis
        auc_range = summary_df['Test AUC'].max() - summary_df['Test AUC'].min()
        print(f"📈 Test AUC Range: {auc_range:.3f} ({summary_df['Test AUC'].min():.3f} - {summary_df['Test AUC'].max():.3f})")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple crosscoders with k-sparse probing")
    parser.add_argument("--config", default="compare_crosscoders.yaml", help="YAML config file")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config with {len(config['crosscoders'])} crosscoders")
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Load LLM (shared across all crosscoders)
    llm_config = config.get('llm', {})
    logger.info(f"Loading LLM: {llm_config.get('base_model_repo', 'default')}")
    
    # Handle null/None values for LoRA model
    base_model = llm_config.get('base_model_repo', "roneneldan/TinyStories-Instruct-33M")
    lora_model = llm_config.get('lora_model_repo', "mars-jason-25/tiny-stories-33M-TSdata-ft1")
    
    # If lora_model_repo is explicitly set to null, use None
    if lora_model == "null" or lora_model is None:
        lora_model = None
    
    # Choose appropriate loading function based on whether we have a LoRA model
    if lora_model is None:
        # Load base model without LoRA
        llm = load_model_with_tl_check(
            model_name=base_model,
            cache_dir=llm_config.get('cache_dir', None),
            device=device,
            dtype=llm_config.get('dtype', "float32"),
        )
    else:
        # Load base model with LoRA
        llm = build_llm_lora(
            base_model_repo=base_model,
            lora_model_repo=lora_model,
            cache_dir=llm_config.get('cache_dir', None),
            device=device,
            dtype=llm_config.get('dtype', None),
        )
    
    # Create classification dataset (shared across all crosscoders)
    eval_config = config['evaluation']
    logger.info(f"Creating {eval_config['dataset']} dataset...")
    texts, labels, label_names = create_classification_dataset(
        dataset_name=eval_config['dataset'],
        n_samples=eval_config['n_samples']
    )
    labels = torch.tensor(labels, device=device)
    
    # Evaluate each crosscoder
    all_results = []
    
    for crosscoder_config in tqdm(config['crosscoders'], desc="Evaluating crosscoders"):
        result = evaluate_single_crosscoder(
            crosscoder_config=crosscoder_config,
            eval_config=eval_config,
            wandb_config=config['wandb'],
            texts=texts,
            labels=labels,
            llm=llm,
            device=device
        )
        all_results.append(result)
    
    # Create comparison tables
    logger.info("Creating comparison tables...")
    comparison_df = create_comparison_table(all_results)
    summary_df = create_summary_table(all_results)
    
    # Print individual crosscoder results first
    print("\n" + "="*80)
    print("INDIVIDUAL CROSSCODER K-SPARSE RESULTS")
    print("="*80)
    
    for result in all_results:
        if result['success']:
            name = result['name']
            lambda_val = result['interaction_penalty']
            
            print(f"\n{name} (λ={lambda_val}):")
            print("-" * (len(name) + len(f" (λ={lambda_val}):") + 1))
            
            # Print compact k results
            for k, k_result in sorted(result['results'].items()):
                train_acc = k_result['train_accuracy']
                test_acc = k_result['test_accuracy']
                print(f"k={k:2d}: {train_acc:.3f}/{test_acc:.3f}")
            
            # Show summary stats for this crosscoder
            summary = result['summary']
            print(f"AUC: {summary['test_auc']:.3f}/{summary['train_auc']:.3f}")
            print(f"Max: {summary['max_test_accuracy']:.3f}/{summary['max_train_accuracy']:.3f}")
        else:
            print(f"\n{result['name']}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Print results
    print_comparison_summary(comparison_df, summary_df)
    
    # Create visualizations
    if not summary_df.empty:
        logger.info("Creating visualizations...")
        create_visualizations(summary_df, output_dir)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config['output']['save_comparison_table']:
        # Save comparison table
        comparison_file = output_dir / f"crosscoder_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file)
        logger.info(f"Comparison table saved to: {comparison_file}")
        
        # Save summary table
        summary_file = output_dir / f"crosscoder_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary table saved to: {summary_file}")
    
    if config['output']['save_individual_results']:
        # Save detailed results
        results_file = output_dir / f"detailed_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to: {results_file}")
    
    return all_results, comparison_df, summary_df


if __name__ == "__main__":
    main()