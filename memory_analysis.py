#!/usr/bin/env python3
"""
Memory analysis of get_neuron_preacts_cutoff function.
"""

import torch

def analyze_preacts_cutoff_memory():
    """Analyze memory usage of the preacts_cutoff function."""
    
    # GPT2-124M dimensions
    batch_size = 128
    hidden_dim = 1536  # crosscoder hidden dim
    d_model = 768      # GPT2 model dim
    d_mlp = 3072       # GPT2 MLP dim  
    n_blocks = 12      # GPT2-124M has 12 layers
    
    print("=== Memory Analysis for get_neuron_preacts_cutoff ===")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Model dim: {d_model}")
    print(f"MLP dim: {d_mlp}")
    print(f"Blocks: {n_blocks}")
    print()
    
    # Input tensors
    print("INPUT TENSORS:")
    enc_acts_size = batch_size * hidden_dim * 4  # float32 = 4 bytes
    print(f"enc_acts_BH: ({batch_size}, {hidden_dim}) = {enc_acts_size / 1e6:.1f} MB")
    
    W_dec_size = n_blocks * hidden_dim * d_model * 4
    print(f"W_dec_PHD: ({n_blocks}, {hidden_dim}, {d_model}) = {W_dec_size / 1e6:.1f} MB")
    
    W_ins_size = n_blocks * d_model * d_mlp * 4
    print(f"W_ins: ({n_blocks}, {d_model}, {d_mlp}) = {W_ins_size / 1e6:.1f} MB")
    print()
    
    # The problematic tensor: W_dec_PBHD
    print("PROBLEMATIC INTERMEDIATE TENSORS:")
    
    # After sorting and indexing: W_dec_PBHD = W_dec_PHD[:,sorted_enc_inds[:,:max_non_zero_index],:]
    # This creates a tensor of shape (n_blocks, batch_size, max_non_zero_index, d_model)
    # In worst case, max_non_zero_index could be close to hidden_dim
    
    max_non_zero_index = hidden_dim  # worst case - all features are non-zero
    W_dec_PBHD_size = n_blocks * batch_size * max_non_zero_index * d_model * 4
    print(f"W_dec_PBHD (worst case): ({n_blocks}, {batch_size}, {max_non_zero_index}, {d_model}) = {W_dec_PBHD_size / 1e9:.2f} GB")
    
    # enc_BHD_W from first einsum
    enc_BHD_W_size = n_blocks * batch_size * d_model * max_non_zero_index * 4
    print(f"enc_BHD_W: ({n_blocks}, {batch_size}, {d_model}, {max_non_zero_index}) = {enc_BHD_W_size / 1e9:.2f} GB")
    
    # Final output p_BNH from second einsum  
    p_BNH_size = n_blocks * batch_size * d_mlp * max_non_zero_index * 4
    print(f"p_BNH: ({n_blocks}, {batch_size}, {d_mlp}, {max_non_zero_index}) = {p_BNH_size / 1e9:.2f} GB")
    
    print()
    print("MEMORY ISSUES:")
    print("1. W_dec_PBHD tensor is HUGE - it indexes the decoder weights for each batch element")
    print("2. The einsum operations create massive intermediate tensors")
    print("3. With sparse activations, most of max_non_zero_index entries are likely zero")
    print()
    print("POTENTIAL SOLUTIONS:")
    print("1. Process in smaller chunks/batches")
    print("2. Use more aggressive sparsity cutoffs") 
    print("3. Implement the computation without the massive indexing operation")
    print("4. Use gradient checkpointing")

if __name__ == "__main__":
    analyze_preacts_cutoff_memory()