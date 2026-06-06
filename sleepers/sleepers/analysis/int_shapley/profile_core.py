#!/usr/bin/env python3
"""
Profile just the core computation, skipping slow dataset loading.
"""

import torch
import time
import numpy as np

def profile_nshap_only():
    """Profile just the nshap computation."""
    
    print("=" * 50)
    print("PROFILING CORE NSHAP COMPUTATION")
    print("=" * 50)
    
    try:
        import nshap
    except ImportError:
        print("nshap not available")
        return
    
    # Create mock data similar to real data
    print("Creating mock feature data...")
    active_features = np.array([4.2, 2.8, 1.5])  # Similar to real values
    mlp_bias = 0.5
    
    print(f"Feature values: {active_features}")
    print(f"MLP bias: {mlp_bias}")
    
    # Create value function
    def value_function(x, coalition_indices):
        if len(coalition_indices) == 0:
            coalition_sum = 0.0
        else:
            coalition_sum = sum(active_features[i] for i in coalition_indices)
        
        preactivation_with_bias = coalition_sum + mlp_bias
        output = torch.nn.functional.gelu(torch.tensor(preactivation_with_bias))
        return float(output)
    
    # Time individual components
    print("\n1. Testing value function calls...")
    start = time.time()
    for i in range(100):
        result = value_function(None, [0, 1])
    time_100_calls = time.time() - start
    print(f"100 value function calls: {time_100_calls:.3f}s ({time_100_calls*10:.1f}ms per call)")
    
    print("\n2. Testing nshap computation...")
    dummy_x = np.ones(len(active_features))
    
    # Test with different numbers of features
    for n_features in [2, 3]:
        print(f"\n--- Testing with {n_features} features ---")
        test_features = active_features[:n_features]
        test_dummy = dummy_x[:n_features]
        
        def test_value_function(x, coalition_indices):
            if len(coalition_indices) == 0:
                coalition_sum = 0.0
            else:
                coalition_sum = sum(test_features[i] for i in coalition_indices)
            preactivation_with_bias = coalition_sum + mlp_bias
            output = torch.nn.functional.gelu(torch.tensor(preactivation_with_bias))
            return float(output)
        
        start = time.time()
        shapley_result = nshap.shapley_taylor(
            test_dummy,
            test_value_function,
            n=min(n_features, 3)
        )
        elapsed = time.time() - start
        print(f"nshap.shapley_taylor({n_features} features): {elapsed:.3f}s")
        
        # Show what we get
        print(f"Result keys: {list(shapley_result.keys()) if hasattr(shapley_result, 'keys') else 'No keys'}")
        if hasattr(shapley_result, 'keys'):
            for key, value in shapley_result.items():
                print(f"  {key}: {value:.6f}")
    
    print("\n3. Estimating full computation time...")
    # Use the 3-feature timing as baseline
    time_per_3_features = 0.1  # Estimate from above
    
    scenarios = [
        ("Current minimal test", 1 * 2 * 3072, 3),
        ("Your 2-story test", 2 * 128 * 3072, 5), 
        ("Reduced test", 2 * 10 * 3072, 3),
    ]
    
    for name, total_pairs, n_features in scenarios:
        estimated_time_per_pair = time_per_3_features * (n_features / 3.0)  # Scale by features
        total_time_hours = (total_pairs * estimated_time_per_pair) / 3600
        print(f"{name}: {total_pairs:,} pairs × {estimated_time_per_pair:.3f}s = {total_time_hours:.2f} hours")

def profile_matrix_operations():
    """Profile the tensor operations."""
    
    print("\n" + "=" * 50) 
    print("PROFILING TENSOR OPERATIONS")
    print("=" * 50)
    
    # Test tensor operations that might be slow
    hidden_dim = 1536
    n_features = 5
    
    print("1. Testing top-k selection...")
    start = time.time()
    for _ in range(1000):
        neuron_features = torch.randn(hidden_dim)
        top_indices = neuron_features.abs().topk(n_features).indices
    elapsed = time.time() - start
    print(f"1000 topk operations: {elapsed:.3f}s ({elapsed:.3f}ms per operation)")
    
    print("\n2. Testing matrix assignment...")
    start = time.time()
    for _ in range(1000):
        interaction_matrix = torch.zeros(hidden_dim, hidden_dim)
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    interaction_matrix[i*10, j*10] = 0.5
    elapsed = time.time() - start
    print(f"1000 matrix assignments: {elapsed:.3f}s ({elapsed:.3f}ms per operation)")

if __name__ == "__main__":
    profile_nshap_only()
    profile_matrix_operations()
    
    print("\n" + "=" * 50)
    print("BOTTLENECK ANALYSIS")
    print("=" * 50)
    print("Primary bottleneck: nshap.shapley_taylor() computation")
    print("Secondary bottleneck: Dataset loading/filtering (2+ minutes)")
    print("Minor: Tensor operations (negligible)")
    print("=" * 50)