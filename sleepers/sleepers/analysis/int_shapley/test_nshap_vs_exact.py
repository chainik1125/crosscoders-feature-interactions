#!/usr/bin/env python3
"""
Test to compare nshap results against exact Shapley-Taylor calculation.
"""

import numpy as np
import torch
import time
from itertools import combinations

def exact_shapley_taylor_pairwise(features, value_func):
    """
    Compute exact pairwise Shapley-Taylor interactions.
    
    For features i,j, the Shapley-Taylor interaction is:
    I(i,j) = Σ over all coalitions S not containing {i,j} of:
             [f(S ∪ {i,j}) - f(S ∪ {i}) - f(S ∪ {j}) + f(S)] * weight(S)
    
    Where weight(S) = |S|! * (n - |S| - 2)! / (n - 1)!
    """
    n_features = len(features)
    interactions = np.zeros((n_features, n_features))
    
    print(f"Computing exact interactions for {n_features} features...")
    
    # For each pair of features
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction_sum = 0.0
            
            # Enumerate all possible coalitions S not containing i or j
            other_features = [k for k in range(n_features) if k != i and k != j]
            
            # Consider all subset sizes of the other features
            for subset_size in range(len(other_features) + 1):
                weight = (
                    np.math.factorial(subset_size) * 
                    np.math.factorial(n_features - subset_size - 2)
                ) / np.math.factorial(n_features - 1)
                
                # Consider all subsets of this size
                if subset_size == 0:
                    subsets = [[]]
                else:
                    subsets = list(combinations(other_features, subset_size))
                
                for subset in subsets:
                    S = list(subset)
                    
                    # Compute the four terms
                    f_S_ij = value_func(None, S + [i, j])
                    f_S_i = value_func(None, S + [i])  
                    f_S_j = value_func(None, S + [j])
                    f_S = value_func(None, S)
                    
                    # Add to interaction sum
                    marginal_interaction = f_S_ij - f_S_i - f_S_j + f_S
                    interaction_sum += weight * marginal_interaction
            
            interactions[i, j] = interaction_sum
            interactions[j, i] = interaction_sum  # Symmetric
    
    return interactions

def create_test_value_function(feature_values, bias=0.0):
    """Create a test value function similar to what we use in practice."""
    
    def value_function(x, coalition_indices):
        if len(coalition_indices) == 0:
            coalition_sum = 0.0
        else:
            coalition_sum = sum(feature_values[i] for i in coalition_indices)
        
        # Apply GELU activation with bias
        preactivation = coalition_sum + bias
        output = torch.nn.functional.gelu(torch.tensor(preactivation, dtype=torch.float32))
        return float(output)
    
    return value_function

def nshap_pairwise(features, value_func):
    """Get pairwise interactions using nshap (our current method)."""
    
    try:
        import nshap
    except ImportError:
        print("nshap not available")
        return None
    
    n_features = len(features)
    dummy_x = np.ones(n_features)
    
    # Compute with nshap
    shapley_result = nshap.shapley_taylor(
        dummy_x,
        value_func,
        n=2  # Only pairwise
    )
    
    # Extract pairwise interactions
    interactions = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                interaction_key = tuple(sorted([i, j]))
                if interaction_key in shapley_result:
                    interactions[i, j] = shapley_result[interaction_key]
    
    return interactions

def test_comparison():
    """Compare nshap vs exact calculation."""
    
    print("=" * 60)
    print("TESTING NSHAP VS EXACT SHAPLEY-TAYLOR CALCULATION")
    print("=" * 60)
    
    # Test different scenarios
    test_cases = [
        {
            'name': 'Small positive features',
            'features': [2.5, 1.8, 0.9],
            'bias': 0.5
        },
        {
            'name': 'Mixed positive/negative',
            'features': [3.2, -1.5, 2.1],
            'bias': 0.0
        },
        {
            'name': 'Large values',
            'features': [8.4, 5.2, -3.1, 1.7],
            'bias': -2.0
        },
        {
            'name': 'Small values near zero',
            'features': [0.1, -0.08, 0.15],
            'bias': 0.02
        }
    ]
    
    all_errors = []
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        features = test_case['features']
        bias = test_case['bias']
        
        print(f"Features: {features}")
        print(f"Bias: {bias}")
        
        # Create value function
        value_func = create_test_value_function(features, bias)
        
        # Test some manual values
        print(f"f(empty): {value_func(None, []):.6f}")
        print(f"f(all): {value_func(None, list(range(len(features)))):.6f}")
        
        # Time exact calculation
        start_time = time.time()
        exact_interactions = exact_shapley_taylor_pairwise(features, value_func)
        exact_time = time.time() - start_time
        
        # Time nshap calculation  
        start_time = time.time()
        nshap_interactions = nshap_pairwise(features, value_func)
        nshap_time = time.time() - start_time
        
        if nshap_interactions is None:
            print("Skipping nshap comparison (not available)")
            continue
        
        # Compare results
        error_matrix = np.abs(exact_interactions - nshap_interactions)
        max_error = np.max(error_matrix)
        mean_error = np.mean(error_matrix[error_matrix > 1e-12])  # Exclude zeros
        
        print(f"Timing:")
        print(f"  Exact: {exact_time:.3f}s")
        print(f"  nshap: {nshap_time:.3f}s")
        print(f"  Speedup: {nshap_time/exact_time:.1f}x slower")
        
        print(f"Results comparison:")
        print(f"  Max absolute error: {max_error:.8f}")
        print(f"  Mean absolute error: {mean_error:.8f}")
        print(f"  Max exact value: {np.max(np.abs(exact_interactions)):.6f}")
        print(f"  Max nshap value: {np.max(np.abs(nshap_interactions)):.6f}")
        
        # Print matrices for small cases
        if len(features) <= 4:
            print("Exact interactions:")
            print(exact_interactions)
            print("nshap interactions:")
            print(nshap_interactions)
            print("Error matrix:")
            print(error_matrix)
        
        all_errors.append(max_error)
        
        # Check if they agree within reasonable tolerance
        if max_error < 1e-6:
            print("✅ EXCELLENT agreement")
        elif max_error < 1e-4:
            print("✅ Good agreement") 
        elif max_error < 1e-2:
            print("⚠️  Moderate agreement")
        else:
            print("❌ Poor agreement")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print(f"=" * 60)
    print(f"Overall max error: {max(all_errors):.8f}")
    print(f"Mean max error: {np.mean(all_errors):.8f}")
    
    if max(all_errors) < 1e-6:
        print("🎉 Exact calculation is highly accurate replacement for nshap!")
    elif max(all_errors) < 1e-4:
        print("✅ Exact calculation is good replacement for nshap")
    else:
        print("⚠️  Significant differences found - investigate further")

if __name__ == "__main__":
    test_comparison()