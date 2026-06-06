#!/usr/bin/env python3
"""
Test to verify that the optimized get_neuron_preacts_cutoff produces identical outputs to the original.
"""

import torch
import sys
import os
import numpy as np

# Add the sleepers module to path
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers')

from sleepers.scripts.utils import get_neuron_preacts_cutoff, add_penalty

def create_original_get_neuron_preacts_cutoff():
    """Create the original version of the function for comparison."""
    
    def original_get_neuron_preacts_cutoff_single_block(enc_acts_BH, W_dec_HD, b_dec_D, W_in, b_in, device="cpu", bias=0):
        """Original implementation for single block (no optimizations)."""
        import einops
        
        hidden_dim = W_dec_HD.shape[0]
        
        # Original sorting logic
        sorted_enc_vals, sorted_enc_inds = torch.sort(torch.abs(enc_acts_BH), dim=-1, descending=True)
        
        # Find max non-zero index
        non_zero_indices = (sorted_enc_vals != 0).sum(dim=1)
        max_non_zero_index = non_zero_indices.max().item()
        
        if max_non_zero_index == 0:
            batch_size = enc_acts_BH.shape[0]
            p_BNH_single = torch.zeros(batch_size, W_in.shape[1], 1, device=enc_acts_BH.device)
            return p_BNH_single, sorted_enc_inds[:, :1]
        
        # Filter to non-zero activations
        filtered_sorted_enc_BH = sorted_enc_vals[:, :max_non_zero_index]
        
        # Original advanced indexing (no gather optimization)
        W_dec_BHD = W_dec_HD[sorted_enc_inds[:, :max_non_zero_index], :]  # Original way
        
        # Original einsum operations
        enc_BHD_W = einops.einsum(filtered_sorted_enc_BH[..., None], W_dec_BHD, 
                                 "batch hidden_c one, batch hidden_c d_model -> batch d_model hidden_c")
        enc_BHD_b = bias * b_dec_D[None, :, None] / hidden_dim
        
        p_BNH_single = einops.einsum(W_in, enc_BHD_W + enc_BHD_b, 
                                    "d_model d_mlp, batch d_model hidden -> batch d_mlp hidden")
        p_BNH_single += bias * b_in[None, :, None] / hidden_dim
        
        return p_BNH_single, sorted_enc_inds
    
    return original_get_neuron_preacts_cutoff_single_block

def test_single_block_equivalence():
    """Test that optimized single block processing matches original."""
    
    print("=== Testing Single Block Equivalence ===")
    
    # Create test data
    torch.manual_seed(42)
    batch_size = 8
    hidden_dim = 64
    d_model = 32
    d_mlp = 128
    
    enc_acts_BH = torch.randn(batch_size, hidden_dim) * 0.5
    # Make some activations zero to test sparsity
    enc_acts_BH[enc_acts_BH.abs() < 0.3] = 0
    
    W_dec_HD = torch.randn(hidden_dim, d_model)
    b_dec_D = torch.randn(d_model)
    W_in = torch.randn(d_model, d_mlp)
    b_in = torch.randn(d_mlp)
    
    # Create tensors for multi-block format
    W_dec_PHD = W_dec_HD.unsqueeze(0)  # (1, hidden, d_model)
    b_dec_PD = b_dec_D.unsqueeze(0)    # (1, d_model)
    W_ins = W_in.unsqueeze(0)          # (1, d_model, d_mlp)
    b_ins = b_in.unsqueeze(0)          # (1, d_mlp)
    W_outs = torch.randn(1, d_mlp, d_model)  # Dummy
    b_outs = torch.randn(1, d_model)         # Dummy
    
    device = "cpu"
    bias = 1.0
    
    # Test original implementation
    original_fn = create_original_get_neuron_preacts_cutoff()
    orig_output, orig_indices = original_fn(enc_acts_BH, W_dec_HD, b_dec_D, W_in, b_in, device, bias)
    
    # Test optimized implementation
    opt_output, opt_indices = get_neuron_preacts_cutoff(
        enc_acts_BH, W_dec_PHD, b_dec_PD, W_ins, b_ins, W_outs, b_outs,
        device=device, bias=bias, block_idx=0
    )
    
    # Compare outputs
    print(f"Original output shape: {orig_output.shape}")
    print(f"Optimized output shape: {opt_output.shape}")
    
    # Check if outputs are close
    output_close = torch.allclose(orig_output, opt_output, rtol=1e-5, atol=1e-6)
    indices_close = torch.equal(orig_indices, opt_indices)
    
    print(f"Outputs match: {output_close}")
    print(f"Indices match: {indices_close}")
    
    if not output_close:
        diff = (orig_output - opt_output).abs()
        print(f"Max absolute difference: {diff.max().item()}")
        print(f"Mean absolute difference: {diff.mean().item()}")
    
    return output_close and indices_close

def test_multi_block_equivalence():
    """Test that the block-by-block processing produces consistent results."""
    
    print("\n=== Testing Multi-Block Equivalence ===")
    
    torch.manual_seed(123)
    batch_size = 4
    hidden_dim = 32
    d_model = 16
    d_mlp = 64
    n_blocks = 3
    
    # Create test data
    enc_acts_BH = torch.randn(batch_size, hidden_dim) * 0.4
    enc_acts_BH[enc_acts_BH.abs() < 0.2] = 0
    
    W_dec_PHD = torch.randn(n_blocks, hidden_dim, d_model)
    b_dec_PD = torch.randn(n_blocks, d_model)
    W_ins = torch.randn(n_blocks, d_model, d_mlp)
    b_ins = torch.randn(n_blocks, d_mlp)
    W_outs = torch.randn(n_blocks, d_mlp, d_model)
    b_outs = torch.randn(n_blocks, d_model)
    
    device = "cpu"
    bias = 1.0
    
    # Test with precomputed sort (optimized path)
    # First compute the sort once
    sorted_enc_vals, sorted_enc_inds = torch.sort(torch.abs(enc_acts_BH), dim=-1, descending=True)
    non_zero_indices = (sorted_enc_vals != 0).sum(dim=1)
    max_non_zero_index = non_zero_indices.max().item()
    precomputed_sort = (sorted_enc_vals, sorted_enc_inds, max_non_zero_index)
    
    # Test each block with and without precomputed sort
    all_match = True
    
    for block_idx in range(n_blocks):
        # Without precomputed sort (computes sort each time)
        output_no_precompute, _ = get_neuron_preacts_cutoff(
            enc_acts_BH, W_dec_PHD, b_dec_PD, W_ins, b_ins, W_outs, b_outs,
            device=device, bias=bias, block_idx=block_idx
        )
        
        # With precomputed sort (optimized)
        output_precompute, _ = get_neuron_preacts_cutoff(
            enc_acts_BH, W_dec_PHD, b_dec_PD, W_ins, b_ins, W_outs, b_outs,
            device=device, bias=bias, block_idx=block_idx, precomputed_sort=precomputed_sort
        )
        
        # Compare
        block_match = torch.allclose(output_no_precompute, output_precompute, rtol=1e-5, atol=1e-6)
        print(f"Block {block_idx} - precomputed sort matches: {block_match}")
        
        if not block_match:
            diff = (output_no_precompute - output_precompute).abs()
            print(f"  Max difference: {diff.max().item()}")
            all_match = False
    
    return all_match

def test_penalty_function_equivalence():
    """Test that add_penalty produces consistent results."""
    
    print("\n=== Testing Penalty Function Equivalence ===")
    
    torch.manual_seed(456)
    batch_size = 6
    hidden_dim = 48
    d_model = 24
    d_mlp = 96
    n_blocks = 4
    
    # Create test data
    enc_acts_BH = torch.randn(batch_size, hidden_dim) * 0.6
    enc_acts_BH[enc_acts_BH.abs() < 0.25] = 0
    
    W_dec_PHD = torch.randn(n_blocks, hidden_dim, d_model)
    b_dec_PD = torch.randn(n_blocks, d_model)
    W_ins = torch.randn(n_blocks, d_model, d_mlp)
    b_ins = torch.randn(n_blocks, d_mlp)
    W_outs = torch.randn(n_blocks, d_mlp, d_model)
    b_outs = torch.randn(n_blocks, d_model)
    
    device = "cpu"
    bias = 1.0
    
    # Define a simple penalty function
    def test_penalty_fn(p_BNH_single):
        return p_BNH_single.abs().mean()
    
    # Test add_penalty function
    penalty_result = add_penalty(
        enc_acts_BH, W_dec_PHD, b_dec_PD, W_ins, b_ins, W_outs, b_outs,
        device, bias, test_penalty_fn
    )
    
    # Manually compute penalty for each block and average (reference implementation)
    manual_penalties = []
    for block_idx in range(n_blocks):
        p_BNH_single, _ = get_neuron_preacts_cutoff(
            enc_acts_BH, W_dec_PHD, b_dec_PD, W_ins, b_ins, W_outs, b_outs,
            device=device, bias=bias, block_idx=block_idx
        )
        manual_penalty = test_penalty_fn(p_BNH_single)
        manual_penalties.append(manual_penalty.item())
    
    manual_mean = sum(manual_penalties) / len(manual_penalties)
    
    print(f"add_penalty result: {penalty_result.item()}")
    print(f"Manual computation: {manual_mean}")
    
    penalty_match = abs(penalty_result.item() - manual_mean) < 1e-6
    print(f"Penalty computation matches: {penalty_match}")
    
    return penalty_match

def run_all_tests():
    """Run all tests and report results."""
    
    print("🧪 Testing Optimized get_neuron_preacts_cutoff Implementation")
    print("=" * 60)
    
    tests = [
        ("Single Block Equivalence", test_single_block_equivalence),
        ("Multi-Block Equivalence", test_multi_block_equivalence),
        ("Penalty Function Equivalence", test_penalty_function_equivalence),
    ]
    
    results = []
    
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n💥 ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Optimizations preserve functionality.")
    else:
        print("\n⚠️  SOME TESTS FAILED! Please review optimizations.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)