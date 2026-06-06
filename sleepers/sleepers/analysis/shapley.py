"""
Shapley value analysis for crosscoder features.

This module implements three key Shapley analysis functions:
1. Neuron-level Shapley values for feature contributions to individual neurons
2. Model output-level Shapley values for feature contributions to model predictions
3. Pairwise Shapley-Taylor interaction indices for feature interactions at neuron level
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable, Any
import itertools
from functools import partial
import warnings

try:
    import nshap
    NSHAP_AVAILABLE = True
except ImportError:
    NSHAP_AVAILABLE = False
    warnings.warn("nshap not available. Install with 'pip install nshap' for Shapley-Taylor functionality.")


def shapley_values_neuron_level(
    preacts_BNH: torch.Tensor,
    postacts_BN: torch.Tensor,
    activation_function: Optional[Callable] = None,
    max_coalition_size: int = 15,
    n_samples: int = 1000
) -> torch.Tensor:
    """
    Compute Shapley values for features contributing to individual neurons.
    
    Args:
        preacts_BNH: Preactivation decomposition [batch, neurons, features]
                    Values sum over H dimension to give preactivation reconstruction
        postacts_BN: Post-activation values [batch, neurons] 
                    Actual or reconstructed neuron activations
        activation_function: Function applied to preactivations to get postactivations
                           If None, assumes identity (postacts = sum(preacts, dim=-1))
        max_coalition_size: Maximum number of features to consider for exact computation
        n_samples: Number of samples for approximation when feature count > max_coalition_size
        
    Returns:
        torch.Tensor: Shapley values [neurons, features]
    """
    B, N, H = preacts_BNH.shape
    device = preacts_BNH.device
    
    # If no activation function provided, assume identity
    if activation_function is None:
        def activation_function(x):
            return x
    
    shapley_values = torch.zeros(N, H, device=device)
    
    for neuron_idx in range(N):
        neuron_preacts = preacts_BNH[:, neuron_idx, :]  # [B, H]
        neuron_postacts = postacts_BN[:, neuron_idx]   # [B]
        
        # Find active features (non-zero across batch)
        active_features = torch.nonzero(neuron_preacts.abs().sum(0) > 1e-8).squeeze(-1)
        
        if len(active_features) == 0:
            continue
            
        if len(active_features) > max_coalition_size:
            # Use sampling approximation
            shapley_vals = _approximate_shapley_neuron(
                neuron_preacts, neuron_postacts, active_features, 
                activation_function, n_samples
            )
        else:
            # Use exact computation
            shapley_vals = _exact_shapley_neuron(
                neuron_preacts, neuron_postacts, active_features, activation_function
            )
        
        shapley_values[neuron_idx, active_features] = shapley_vals
    
    return shapley_values


def _exact_shapley_neuron(
    preacts_BH: torch.Tensor,
    postacts_B: torch.Tensor, 
    active_features: torch.Tensor,
    activation_function: Callable
) -> torch.Tensor:
    """Exact Shapley computation for small feature sets."""
    H = len(active_features)
    shapley_vals = torch.zeros(H, device=preacts_BH.device)
    
    for i, feature_idx in enumerate(active_features):
        marginal_contributions = []
        
        # Iterate over all coalitions not containing current feature
        for r in range(H):
            for coalition in itertools.combinations(range(H), r):
                if i in coalition:
                    continue
                
                # Coalition output
                coalition_mask = torch.zeros(H, device=preacts_BH.device)
                coalition_mask[list(coalition)] = 1
                coalition_preacts = (preacts_BH[:, active_features] * coalition_mask).sum(-1)
                coalition_output = activation_function(coalition_preacts).mean()
                
                # Coalition + current feature output
                extended_mask = coalition_mask.clone()
                extended_mask[i] = 1
                extended_preacts = (preacts_BH[:, active_features] * extended_mask).sum(-1)
                extended_output = activation_function(extended_preacts).mean()
                
                marginal_contribution = extended_output - coalition_output
                marginal_contributions.append(marginal_contribution)
        
        shapley_vals[i] = torch.stack(marginal_contributions).mean()
    
    return shapley_vals


def _approximate_shapley_neuron(
    preacts_BH: torch.Tensor,
    postacts_B: torch.Tensor,
    active_features: torch.Tensor, 
    activation_function: Callable,
    n_samples: int
) -> torch.Tensor:
    """Sampling approximation for large feature sets."""
    H = len(active_features)
    shapley_vals = torch.zeros(H, device=preacts_BH.device)
    
    for i, feature_idx in enumerate(active_features):
        marginal_contributions = []
        
        for _ in range(n_samples):
            # Random coalition size and members (excluding current feature)
            coalition_size = torch.randint(0, H, (1,)).item()
            available_features = torch.cat([
                torch.arange(i, device=preacts_BH.device),
                torch.arange(i+1, H, device=preacts_BH.device)
            ])
            
            if len(available_features) == 0:
                coalition = torch.tensor([], device=preacts_BH.device, dtype=torch.long)
            else:
                coalition_size = min(coalition_size, len(available_features))
                coalition = available_features[torch.randperm(len(available_features))[:coalition_size]]
            
            # Coalition output
            coalition_mask = torch.zeros(H, device=preacts_BH.device)
            coalition_mask[coalition] = 1
            coalition_preacts = (preacts_BH[:, active_features] * coalition_mask).sum(-1)
            coalition_output = activation_function(coalition_preacts).mean()
            
            # Coalition + current feature output
            extended_mask = coalition_mask.clone()
            extended_mask[i] = 1
            extended_preacts = (preacts_BH[:, active_features] * extended_mask).sum(-1)
            extended_output = activation_function(extended_preacts).mean()
            
            marginal_contribution = extended_output - coalition_output
            marginal_contributions.append(marginal_contribution)
        
        shapley_vals[i] = torch.stack(marginal_contributions).mean()
    
    return shapley_vals


def shapley_values_output_level(
    encoding_BH: torch.Tensor,
    output_function: Callable[[torch.Tensor], torch.Tensor],
    max_coalition_size: int = 15,
    n_samples: int = 1000
) -> torch.Tensor:
    """
    Compute Shapley values for features contributing to model output.
    
    Args:
        encoding_BH: Feature encodings [batch, features]
        output_function: Function that takes encoding and returns scalar output per batch
        max_coalition_size: Maximum number of features for exact computation
        n_samples: Number of samples for approximation
        
    Returns:
        torch.Tensor: Shapley values [features]
    """
    B, H = encoding_BH.shape
    device = encoding_BH.device
    
    # Find active features
    active_features = torch.nonzero(encoding_BH.abs().sum(0) > 1e-8).squeeze(-1)
    
    if len(active_features) == 0:
        return torch.zeros(H, device=device)
    
    if len(active_features) > max_coalition_size:
        # Use sampling approximation
        shapley_vals = _approximate_shapley_output(
            encoding_BH, output_function, active_features, n_samples
        )
    else:
        # Use exact computation
        shapley_vals = _exact_shapley_output(
            encoding_BH, output_function, active_features
        )
    
    # Map back to full feature space
    full_shapley = torch.zeros(H, device=device)
    full_shapley[active_features] = shapley_vals
    
    return full_shapley


def _exact_shapley_output(
    encoding_BH: torch.Tensor,
    output_function: Callable,
    active_features: torch.Tensor
) -> torch.Tensor:
    """Exact Shapley computation for output level."""
    H = len(active_features)
    shapley_vals = torch.zeros(H, device=encoding_BH.device)
    
    for i, feature_idx in enumerate(active_features):
        marginal_contributions = []
        
        # Iterate over all coalitions not containing current feature
        for r in range(H):
            for coalition in itertools.combinations(range(H), r):
                if i in coalition:
                    continue
                
                # Coalition output
                coalition_mask = torch.zeros(H, device=encoding_BH.device)
                coalition_mask[list(coalition)] = 1
                masked_encoding = encoding_BH[:, active_features] * coalition_mask
                coalition_output = output_function(masked_encoding).mean()
                
                # Coalition + current feature output
                extended_mask = coalition_mask.clone()
                extended_mask[i] = 1
                extended_encoding = encoding_BH[:, active_features] * extended_mask
                extended_output = output_function(extended_encoding).mean()
                
                marginal_contribution = extended_output - coalition_output
                marginal_contributions.append(marginal_contribution)
        
        shapley_vals[i] = torch.stack(marginal_contributions).mean()
    
    return shapley_vals


def _approximate_shapley_output(
    encoding_BH: torch.Tensor,
    output_function: Callable,
    active_features: torch.Tensor,
    n_samples: int
) -> torch.Tensor:
    """Sampling approximation for output-level Shapley values."""
    H = len(active_features)
    shapley_vals = torch.zeros(H, device=encoding_BH.device)
    
    for i, feature_idx in enumerate(active_features):
        marginal_contributions = []
        
        for _ in range(n_samples):
            # Random coalition size and members (excluding current feature)
            coalition_size = torch.randint(0, H, (1,)).item()
            available_features = torch.cat([
                torch.arange(i, device=encoding_BH.device),
                torch.arange(i+1, H, device=encoding_BH.device)
            ])
            
            if len(available_features) == 0:
                coalition = torch.tensor([], device=encoding_BH.device, dtype=torch.long)
            else:
                coalition_size = min(coalition_size, len(available_features))
                coalition = available_features[torch.randperm(len(available_features))[:coalition_size]]
            
            # Coalition output
            coalition_mask = torch.zeros(H, device=encoding_BH.device)
            coalition_mask[coalition] = 1
            masked_encoding = encoding_BH[:, active_features] * coalition_mask
            coalition_output = output_function(masked_encoding).mean()
            
            # Coalition + current feature output
            extended_mask = coalition_mask.clone()
            extended_mask[i] = 1
            extended_encoding = encoding_BH[:, active_features] * extended_mask
            extended_output = output_function(extended_encoding).mean()
            
            marginal_contribution = extended_output - coalition_output
            marginal_contributions.append(marginal_contribution)
        
        shapley_vals[i] = torch.stack(marginal_contributions).mean()
    
    return shapley_vals


def shapley_taylor_pairwise_neuron(
    preacts_BNH: torch.Tensor,
    postacts_BN: torch.Tensor,
    neuron_idx: int,
    activation_function: Optional[Callable] = None,
    max_features: int = 10,
    num_samples: int = 2000,
    mlp_bias: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute pairwise Shapley-Taylor interaction indices for a single neuron.
    
    Args:
        preacts_BNH: Preactivation decomposition [batch, neurons, features]
        postacts_BN: Post-activation values [batch, neurons]
        neuron_idx: Index of target neuron
        activation_function: Function applied to preactivations
        max_features: Maximum number of features to consider
        num_samples: Number of samples for nshap computation
        
    Returns:
        Tuple of:
            - torch.Tensor: Pairwise interaction matrix [features, features]
            - Dict: Metadata including individual Shapley values
    """
    if not NSHAP_AVAILABLE:
        raise ImportError("nshap library required for Shapley-Taylor computation. Install with 'pip install nshap'")
    
    B, N, H = preacts_BNH.shape
    device = preacts_BNH.device
    
    if activation_function is None:
        def activation_function(x):
            return x
    
    # Extract data for target neuron
    neuron_preacts = preacts_BNH[:, neuron_idx, :]  # [B, H]
    neuron_postacts = postacts_BN[:, neuron_idx]   # [B]
    
    # Find active features
    active_features = torch.nonzero(neuron_preacts.abs().sum(0) > 1e-8).squeeze(-1)
    
    if len(active_features) == 0:
        return torch.zeros(H, H), {'active_features': [], 'individual_shapley': torch.zeros(H)}
    
    # Optionally limit features for computational tractability (0 = no limit)
    if max_features > 0 and len(active_features) > max_features:
        # Select top features by activation magnitude
        feature_importance = neuron_preacts.abs().sum(0)
        top_features = feature_importance.topk(max_features).indices
        active_features = active_features[torch.isin(active_features, top_features)]
    
    active_features_np = active_features.cpu().numpy()
    n_features = len(active_features_np)
    
    # Get MLP bias for this neuron if provided
    neuron_bias = 0.0
    if mlp_bias is not None:
        neuron_bias = float(mlp_bias[neuron_idx].cpu())
    
    # Create value function for nshap
    def value_function(x: np.ndarray, coalition_indices: list) -> float:
        """
        Value function that takes datapoint and coalition indices, returns neuron output.
        x: datapoint (ignored - we use precomputed values)
        coalition_indices: list of feature indices that are active
        """
        # Create mask from coalition indices
        mask_tensor = torch.zeros(n_features, dtype=torch.float32, device=device)
        if len(coalition_indices) > 0:
            mask_tensor[coalition_indices] = 1.0
        
        # Apply mask to preactivations
        masked_preacts = neuron_preacts[:, active_features] * mask_tensor
        summed_preacts = masked_preacts.sum(-1)  # [B]
        
        # Add MLP bias to the preactivation sum
        summed_preacts_with_bias = summed_preacts + neuron_bias
        
        # Apply activation function and return mean
        output = activation_function(summed_preacts_with_bias).mean()
        return float(output.cpu())
    
    # Create dummy data point (values don't matter for our use case)
    dummy_x = np.ones(n_features)
    
    # Compute Shapley-Taylor interaction index
    try:
        shapley_taylor = nshap.shapley_taylor(
            dummy_x,
            value_function,
            n=min(n_features, 10)  # Limit n to avoid excessive computation
        )
    except Exception as e:
        warnings.warn(f"nshap computation failed: {e}. Returning zero matrix.")
        return torch.zeros(H, H), {'active_features': active_features_np, 'individual_shapley': torch.zeros(H)}
    
    # Extract pairwise interaction matrix
    interaction_matrix = torch.zeros(H, H, device=device)
    
    # Fill in pairwise interactions
    for i, feat_i in enumerate(active_features_np):
        for j, feat_j in enumerate(active_features_np):
            if i != j:
                # Get pairwise interaction value - shapley_taylor is dict-like
                interaction_key = tuple(sorted([i, j]))
                if interaction_key in shapley_taylor:
                    interaction_matrix[feat_i, feat_j] = shapley_taylor[interaction_key]
    
    # Extract individual Shapley values (main effects)
    individual_shapley = torch.zeros(H, device=device)
    for i, feat_idx in enumerate(active_features_np):
        if (i,) in shapley_taylor:
            individual_shapley[feat_idx] = shapley_taylor[(i,)]
    
    metadata = {
        'active_features': active_features_np,
        'individual_shapley': individual_shapley,
        'neuron_idx': neuron_idx,
        'n_samples': num_samples,
        'shapley_taylor_object': shapley_taylor
    }
    
    return interaction_matrix, metadata


# Convenience wrapper functions
def analyze_neuron_shapley(
    preacts_BNH: torch.Tensor,
    postacts_BN: torch.Tensor,
    neuron_indices: Optional[List[int]] = None,
    activation_function: Optional[Callable] = None,
    **kwargs
) -> Dict[int, torch.Tensor]:
    """
    Analyze Shapley values for multiple neurons.
    
    Returns:
        Dict mapping neuron_idx -> shapley_values [features]
    """
    _, N, _ = preacts_BNH.shape
    
    if neuron_indices is None:
        neuron_indices = list(range(N))
    
    results = {}
    full_shapley = shapley_values_neuron_level(preacts_BNH, postacts_BN, activation_function, **kwargs)
    
    for neuron_idx in neuron_indices:
        results[neuron_idx] = full_shapley[neuron_idx]
    
    return results


def analyze_neuron_interactions(
    preacts_BNH: torch.Tensor,
    postacts_BN: torch.Tensor,
    neuron_indices: Optional[List[int]] = None,
    activation_function: Optional[Callable] = None,
    **kwargs
) -> Dict[int, Tuple[torch.Tensor, Dict]]:
    """
    Analyze Shapley-Taylor interactions for multiple neurons.
    
    Returns:
        Dict mapping neuron_idx -> (interaction_matrix, metadata)
    """
    _, N, _ = preacts_BNH.shape
    
    if neuron_indices is None:
        neuron_indices = list(range(min(5, N)))  # Default to first 5 neurons
    
    results = {}
    for neuron_idx in neuron_indices:
        interaction_matrix, metadata = shapley_taylor_pairwise_neuron(
            preacts_BNH, postacts_BN, neuron_idx, activation_function, **kwargs
        )
        results[neuron_idx] = (interaction_matrix, metadata)
    
    return results


def shapley_values_dominant_feature_neuron_level(
    preacts_BNH: torch.Tensor,
    postacts_BN: torch.Tensor,
    activation_function: Optional[Callable] = None,
    max_coalition_size: int = 15,
    n_samples: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Shapley values for dominant features vs remaining features at each token.
    
    For each (batch, neuron) position, features are ordered by their absolute value,
    but retain their signed values. We compute:
    1. Shapley value of the dominant (largest abs value) feature
    2. Shapley value of all remaining features combined
    
    Args:
        preacts_BNH: Preactivation decomposition [batch, neurons, features]
                    Values sum over H dimension to give preactivation reconstruction  
        postacts_BN: Post-activation values [batch, neurons]
                    Actual or reconstructed neuron activations
        activation_function: Function applied to preactivations to get postactivations
                           If None, assumes identity (postacts = sum(preacts, dim=-1))
        max_coalition_size: Maximum number of features to consider for exact computation
        n_samples: Number of samples for approximation when feature count > max_coalition_size
        
    Returns:
        Tuple of:
            - torch.Tensor: Shapley values for dominant feature [batch, neurons]
            - torch.Tensor: Shapley values for remaining features [batch, neurons] 
            - torch.Tensor: Indices of dominant features [batch, neurons]
    """
    B, N, H = preacts_BNH.shape
    device = preacts_BNH.device
    
    if activation_function is None:
        def activation_function(x):
            return x
    
    # Find dominant feature for each (batch, neuron) position
    abs_preacts = preacts_BNH.abs()  # [B, N, H]
    dominant_indices = abs_preacts.argmax(dim=-1)  # [B, N]
    
    # Initialize output tensors
    dominant_shapley = torch.zeros(B, N, device=device)
    remaining_shapley = torch.zeros(B, N, device=device)
    
    for b in range(B):
        for n in range(N):
            dominant_feat_idx = dominant_indices[b, n].item()
            
            # Get preactivations and postactivation for this (batch, neuron)
            token_preacts = preacts_BNH[b, n, :]  # [H]
            token_postact = postacts_BN[b, n]     # scalar
            
            # Skip if all preactivations are essentially zero
            if token_preacts.abs().max() < 1e-8:
                continue
            
            # Compute 2-player Shapley game: dominant vs remaining
            dominant_value = token_preacts[dominant_feat_idx]
            
            # Create remaining features mask (all except dominant)
            remaining_mask = torch.ones(H, dtype=torch.bool, device=device)
            remaining_mask[dominant_feat_idx] = False
            remaining_values = token_preacts[remaining_mask]
            remaining_sum = remaining_values.sum()
            
            # 2-player Shapley computation
            # Marginal contributions:
            # 1. Add dominant to empty coalition
            empty_output = activation_function(torch.tensor(0.0, device=device))
            dominant_alone_output = activation_function(dominant_value)
            dominant_marginal_1 = dominant_alone_output - empty_output
            
            # 2. Add dominant to remaining coalition  
            remaining_alone_output = activation_function(remaining_sum)
            both_output = activation_function(dominant_value + remaining_sum)
            dominant_marginal_2 = both_output - remaining_alone_output
            
            # 3. Add remaining to empty coalition
            remaining_marginal_1 = remaining_alone_output - empty_output
            
            # 4. Add remaining to dominant coalition
            remaining_marginal_2 = both_output - dominant_alone_output
            
            # Shapley values are average of marginal contributions
            dominant_shapley[b, n] = (dominant_marginal_1 + dominant_marginal_2) / 2
            remaining_shapley[b, n] = (remaining_marginal_1 + remaining_marginal_2) / 2
    
    return dominant_shapley, remaining_shapley, dominant_indices


def shapley_values_top_k_features_neuron_level(
    preacts_BNH: torch.Tensor,
    postacts_BN: torch.Tensor,
    k: int = 3,
    activation_function: Optional[Callable] = None,
    max_coalition_size: int = 15,
    n_samples: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Shapley values for top-k features (by absolute value) vs remaining features.
    
    For each (batch, neuron) position, features are ordered by their absolute value,
    but retain their signed values. We compute Shapley values for:
    1. Each of the top-k features individually
    2. All remaining features combined as one coalition
    
    Args:
        preacts_BNH: Preactivation decomposition [batch, neurons, features]
        postacts_BN: Post-activation values [batch, neurons] 
        k: Number of top features to analyze individually
        activation_function: Function applied to preactivations
        max_coalition_size: Maximum number of features for exact computation
        n_samples: Number of samples for approximation
        
    Returns:
        Tuple of:
            - torch.Tensor: Shapley values for top-k features [batch, neurons, k]
            - torch.Tensor: Shapley values for remaining features [batch, neurons]
            - torch.Tensor: Indices of top-k features [batch, neurons, k]
    """
    B, N, H = preacts_BNH.shape
    device = preacts_BNH.device
    
    if activation_function is None:
        def activation_function(x):
            return x
    
    # Find top-k features by absolute value for each (batch, neuron) position
    abs_preacts = preacts_BNH.abs()  # [B, N, H]
    _, top_k_indices = torch.topk(abs_preacts, k=min(k, H), dim=-1)  # [B, N, k]
    
    # Initialize output tensors
    top_k_shapley = torch.zeros(B, N, k, device=device)
    remaining_shapley = torch.zeros(B, N, device=device)
    
    for b in range(B):
        for n in range(N):
            # Get preactivations for this (batch, neuron)
            token_preacts = preacts_BNH[b, n, :]  # [H]
            token_postact = postacts_BN[b, n]     # scalar
            
            # Skip if all preactivations are essentially zero
            if token_preacts.abs().max() < 1e-8:
                continue
            
            # Get top-k feature indices and values
            top_k_feat_indices = top_k_indices[b, n, :]  # [k]
            top_k_values = token_preacts[top_k_feat_indices]  # [k]
            
            # Get remaining features mask and values
            remaining_mask = torch.ones(H, dtype=torch.bool, device=device)
            remaining_mask[top_k_feat_indices] = False
            remaining_values = token_preacts[remaining_mask]
            remaining_sum = remaining_values.sum()
            
            # Compute (k+1)-player Shapley game
            n_players = k + 1  # k individual features + 1 remaining coalition
            
            if n_players <= max_coalition_size:
                # Exact computation
                shapley_vals = _exact_shapley_top_k(
                    top_k_values, remaining_sum, activation_function
                )
            else:
                # Sampling approximation
                shapley_vals = _approximate_shapley_top_k(
                    top_k_values, remaining_sum, activation_function, n_samples
                )
            
            top_k_shapley[b, n, :] = shapley_vals[:k]
            remaining_shapley[b, n] = shapley_vals[k]
    
    return top_k_shapley, remaining_shapley, top_k_indices


def _exact_shapley_top_k(
    top_k_values: torch.Tensor,
    remaining_sum: torch.Tensor,
    activation_function: Callable
) -> torch.Tensor:
    """Exact Shapley computation for top-k + remaining coalition game."""
    k = len(top_k_values)
    n_players = k + 1
    shapley_vals = torch.zeros(n_players, device=top_k_values.device)
    
    # All possible coalitions of size 0 to k
    for coalition_size in range(n_players):
        for coalition in itertools.combinations(range(n_players), coalition_size):
            coalition_set = set(coalition)
            
            # Compute coalition value
            coalition_value = torch.tensor(0.0, device=top_k_values.device)
            for player_idx in coalition:
                if player_idx < k:  # Top-k feature
                    coalition_value += top_k_values[player_idx]
                else:  # Remaining coalition
                    coalition_value += remaining_sum
            
            coalition_output = activation_function(coalition_value)
            
            # Compute marginal contribution for each player not in coalition
            for player_idx in range(n_players):
                if player_idx not in coalition_set:
                    # Add this player to coalition
                    extended_value = coalition_value.clone()
                    if player_idx < k:  # Top-k feature
                        extended_value += top_k_values[player_idx]
                    else:  # Remaining coalition
                        extended_value += remaining_sum
                    
                    extended_output = activation_function(extended_value)
                    marginal_contribution = extended_output - coalition_output
                    
                    # Weight by coalition size probability
                    weight = 1.0 / (n_players * comb(n_players - 1, coalition_size))
                    shapley_vals[player_idx] += weight * marginal_contribution
    
    return shapley_vals


def _approximate_shapley_top_k(
    top_k_values: torch.Tensor,
    remaining_sum: torch.Tensor, 
    activation_function: Callable,
    n_samples: int
) -> torch.Tensor:
    """Sampling approximation for top-k + remaining coalition game."""
    k = len(top_k_values)
    n_players = k + 1
    shapley_vals = torch.zeros(n_players, device=top_k_values.device)
    
    for player_idx in range(n_players):
        marginal_contributions = []
        
        for _ in range(n_samples):
            # Sample random coalition not containing current player
            other_players = [i for i in range(n_players) if i != player_idx]
            coalition_size = torch.randint(0, len(other_players) + 1, (1,)).item()
            
            if coalition_size > 0:
                coalition = torch.randperm(len(other_players))[:coalition_size]
                coalition_players = [other_players[i] for i in coalition]
            else:
                coalition_players = []
            
            # Compute coalition value
            coalition_value = torch.tensor(0.0, device=top_k_values.device)
            for p_idx in coalition_players:
                if p_idx < k:  # Top-k feature
                    coalition_value += top_k_values[p_idx]
                else:  # Remaining coalition
                    coalition_value += remaining_sum
            
            coalition_output = activation_function(coalition_value)
            
            # Add current player to coalition
            extended_value = coalition_value.clone()
            if player_idx < k:  # Top-k feature
                extended_value += top_k_values[player_idx]
            else:  # Remaining coalition
                extended_value += remaining_sum
            
            extended_output = activation_function(extended_value)
            marginal_contribution = extended_output - coalition_output
            marginal_contributions.append(marginal_contribution)
        
        shapley_vals[player_idx] = torch.stack(marginal_contributions).mean()
    
    return shapley_vals


def comb(n: int, k: int) -> int:
    """Compute binomial coefficient n choose k."""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    
    # Use multiplicative formula to avoid large factorials
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result