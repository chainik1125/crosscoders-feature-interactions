#!/usr/bin/env python3
"""Debug by comparing crosscoder decomposition vs real MLP activations."""

import sys
import os
import torch

# Add paths  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from compare_with_config import load_config, setup_device, load_models_and_data
from sleepers.analysis.analysis_utils import get_activations, get_preacts_nocontract

def compare_real_vs_reconstructed():
    """Compare real MLP activations vs crosscoder reconstruction."""
    print("=" * 60)
    print("COMPARING REAL vs CROSSCODER ACTIVATIONS")
    print("=" * 60)
    
    # Load config and models
    config = load_config("conservative_test.yaml")
    device = setup_device(config)
    dataset, llm, crosscoder = load_models_and_data(config, device)
    
    # Get data for one story
    story_text = dataset[0]['text']
    
    # Get crosscoder activations
    feature_activations_SH, activations_SMLD = get_activations(story_text, llm, crosscoder)
    
    # Get crosscoder preactivation reconstruction
    preacts = get_preacts_nocontract(
        feature_activations_SH,
        crosscoder.W_dec_HXD,
        crosscoder.b_dec_XD,
        llm,
        block=0,
        bias=True
    )
    
    # Sum crosscoder reconstruction
    crosscoder_preacts = preacts.sum(dim=0)  # [d_mlp, hidden_dim] -> [d_mlp] 
    crosscoder_reconstruction = crosscoder_preacts.sum(dim=-1)  # [d_mlp]
    
    # Get REAL MLP activations from the model
    # activations_SMLD shape: [seq_len, model, layer, d_model]
    real_mlp_input = activations_SMLD[:, 0, 0, :]  # [seq_len, d_model]
    
    # Pass through MLP to get real preactivations
    W_in = llm.blocks[0].mlp.W_in  # [d_model, d_mlp]
    b_in = llm.blocks[0].mlp.b_in  # [d_mlp]
    
    real_preacts = torch.einsum('sd,dm->sm', real_mlp_input, W_in) + b_in  # [seq_len, d_mlp]
    real_postacts = torch.relu(real_preacts)  # [seq_len, d_mlp]
    
    # Sum over sequence length for comparison
    real_preacts_sum = real_preacts.sum(dim=0)  # [d_mlp]
    real_postacts_sum = real_postacts.sum(dim=0)  # [d_mlp]
    
    print(f"Shapes:")
    print(f"  Crosscoder reconstruction: {crosscoder_reconstruction.shape}")
    print(f"  Real MLP preacts: {real_preacts_sum.shape}")
    print(f"  Real MLP postacts: {real_postacts_sum.shape}")
    
    # Compare statistics
    print(f"\nCrosscoder reconstruction (sum over sequence):")
    print(f"  Min: {crosscoder_reconstruction.min():.6f}")
    print(f"  Max: {crosscoder_reconstruction.max():.6f}")
    print(f"  Mean: {crosscoder_reconstruction.mean():.6f}")
    print(f"  Positive count: {(crosscoder_reconstruction > 0).sum()}")
    
    print(f"\nReal MLP preactivations (sum over sequence):")
    print(f"  Min: {real_preacts_sum.min():.6f}")
    print(f"  Max: {real_preacts_sum.max():.6f}")
    print(f"  Mean: {real_preacts_sum.mean():.6f}")  
    print(f"  Positive count: {(real_preacts_sum > 0).sum()}")
    
    print(f"\nReal MLP postactivations (sum over sequence):")
    print(f"  Min: {real_postacts_sum.min():.6f}")
    print(f"  Max: {real_postacts_sum.max():.6f}")
    print(f"  Mean: {real_postacts_sum.mean():.6f}")
    print(f"  Positive count: {(real_postacts_sum > 0).sum()}")
    
    # Check correlation
    correlation = torch.corrcoef(torch.stack([crosscoder_reconstruction.cpu(), real_preacts_sum.cpu()]))[0, 1]
    print(f"\nCorrelation (crosscoder vs real preacts): {correlation:.4f}")
    
    # Find neurons that are positive in real but negative in crosscoder
    real_positive = real_preacts_sum > 0
    crosscoder_negative = crosscoder_reconstruction < 0
    mismatch = real_positive & crosscoder_negative
    
    print(f"\nMismatch analysis:")
    print(f"  Real positive, crosscoder negative: {mismatch.sum()} neurons")
    
    if mismatch.sum() > 0:
        mismatch_indices = torch.nonzero(mismatch).squeeze(-1)[:5]  # Show first 5
        print(f"  Examples:")
        for idx in mismatch_indices:
            idx = idx.item()
            print(f"    Neuron {idx}: real={real_preacts_sum[idx]:.6f}, crosscoder={crosscoder_reconstruction[idx]:.6f}")
    
    return crosscoder_reconstruction, real_preacts_sum, real_postacts_sum

if __name__ == "__main__":
    result = compare_real_vs_reconstructed()