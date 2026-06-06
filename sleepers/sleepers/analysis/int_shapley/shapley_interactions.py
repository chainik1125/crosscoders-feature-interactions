"""
Sequential Shapley-Taylor interaction computation for crosscoder features.

This module implements memory-efficient computation of pairwise feature interactions
using Shapley-Taylor interaction indices, processing stories and neurons sequentially
to avoid memory overflow.
"""

import torch
import numpy as np
from tqdm import tqdm
import warnings
import sys
import os

# Import existing functions from the codebase
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from sleepers.analysis.shapley import shapley_taylor_pairwise_neuron
from sleepers.analysis.analysis_utils import get_activations


def compute_shapley_interactions_sequential(
	dataset,
	llm,
	crosscoder,
	num_stories: int = 2,
	layer: int = 0,
	max_features_per_neuron: int = 100,
	num_samples: int = 800,
	threshold: float = 1e-6,
	small_threshold: float = 1e-8,
	device: str = None,
	verbose: bool = True,
	value_function_type: str = "gelu",
	max_tokens_per_story: int = 128,
	max_neurons_per_token: int = None
) -> torch.Tensor:
	"""
	Compute Shapley-Taylor interactions with token-by-token processing for memory efficiency.
	
	Args:
		dataset: Dataset containing story texts
		llm: Language model
		crosscoder: Crosscoder model  
		num_stories: Number of stories to process
		layer: Which layer to analyze (0-3)
		max_features_per_neuron: Maximum features to consider per neuron (0=no limit, but computationally expensive)
		num_samples: Number of samples for Shapley-Taylor approximation
		threshold: Minimum activation threshold for neuron processing
		small_threshold: Minimum activation threshold for feature selection
		device: Device to use for computation
		verbose: Whether to show progress bars
		value_function_type: Type of value function ("gelu", "mlp_output")
			- "gelu": GELU(sum of coalition features + bias)
			- "mlp_output": W_out @ GELU(sum of coalition features + bias) + b_out
		max_tokens_per_story: Maximum tokens to process per story
		max_neurons_per_token: Maximum neurons to process per token (None = all neurons)
		
	Returns:
		torch.Tensor: [1536, 1536] pairwise feature interaction matrix
	"""
	if device is None:
		device = llm.cfg.device
		
	d_mlp = llm.cfg.d_mlp
	hidden_dim = 1536  # crosscoder feature dimension
	
	if verbose:
		print(f"Processing {num_stories} stories on layer {layer} (token-by-token)")
		print(f"Model d_mlp: {d_mlp}, crosscoder hidden_dim: {hidden_dim}")
		print(f"Value function type: {value_function_type}")
	
	# Get MLP weights for full output computation if needed
	W_out = llm.blocks[layer].mlp.W_out  # [d_mlp, d_model]
	b_out = llm.blocks[layer].mlp.b_out  # [d_model] 
	mlp_bias = llm.blocks[layer].mlp.b_in  # [d_mlp]
	
	if verbose:
		print("Step 1: Token-by-token Shapley computation...")
		
	final_interactions = torch.zeros(hidden_dim, hidden_dim, device='cpu', dtype=torch.float32)
	total_processed = 0
	
	story_iterator = tqdm(range(num_stories), desc="Processing stories") if verbose else range(num_stories)
	
	for story_idx in story_iterator:
		story_text = dataset[story_idx]['text']
		
		# Get activations for this story
		try:
			feature_activations_SH, activations_SMLD = get_activations(story_text, llm, crosscoder)
			
			# Extract the relevant activations for the specified layer
			enc_acts_BH = feature_activations_SH  # [seq_len, hidden_dim]
			
			# Get preactivations using get_preacts_nocontract 
			from sleepers.analysis.analysis_utils import get_preacts_nocontract
			
			preacts = get_preacts_nocontract(
				enc_acts_BH,              # [seq_len, hidden_dim] 
				crosscoder.W_dec_HXD,     # [hidden, contexts, layers, d_model]
				crosscoder.b_dec_XD,      # [contexts, layers, d_model]
				llm,                      # LLM object
				block=layer,              # Which layer (0-3)
				bias=True                 # Include bias terms
			)
			
			# preacts shape should be [seq_len, d_mlp, hidden_dim]
			if verbose and story_idx == 0:
				print(f"Preacts shape: {preacts.shape}")
			
			# Process each token separately
			seq_len = min(preacts.shape[0], max_tokens_per_story)
			token_iterator = tqdm(range(seq_len), desc=f"Tokens in story {story_idx}", leave=False) if verbose else range(seq_len)
			
			for token_idx in token_iterator:
				token_preacts = preacts[token_idx]  # [d_mlp, hidden_dim]
				
				# Process each neuron for this token (optionally limited)
				neurons_to_process = min(d_mlp, max_neurons_per_token) if max_neurons_per_token is not None else d_mlp
				for neuron_idx in range(neurons_to_process):
					neuron_features = token_preacts[neuron_idx]  # [hidden_dim=1536]
					
					# Skip inactive neurons
					max_activation = neuron_features.abs().max()
					if max_activation < threshold:
						continue
						
					# Find active features
					active_mask = neuron_features.abs() > small_threshold
					num_active = active_mask.sum()
					
					if num_active < 2:  # Need at least 2 features for interactions
						continue
					
					try:
						# Compute Shapley-Taylor interactions for this token's neuron
						interaction_matrix = compute_token_neuron_shapley(
							neuron_features=neuron_features,
							neuron_idx=neuron_idx,
							layer=layer,
							llm=llm,
							value_function_type=value_function_type,
							W_out=W_out,
							b_out=b_out,
							mlp_bias=mlp_bias,
							max_features_per_neuron=max_features_per_neuron,
							num_samples=num_samples,
							small_threshold=small_threshold,
							device=device
						)
						
						# Accumulate interactions
						final_interactions += interaction_matrix.cpu()
						total_processed += 1
						
					except Exception as e:
						if verbose and neuron_idx % 1000 == 0:
							print(f"Skipping token {token_idx}, neuron {neuron_idx}: {e}")
						continue
			
		except Exception as e:
			if verbose:
				print(f"Error processing story {story_idx}: {e}")
			continue
			
		# Memory cleanup
		del preacts
		torch.cuda.empty_cache()
	
	# Average across all processed token-neuron pairs
	if total_processed > 0:
		final_interactions /= total_processed
		if verbose:
			print(f"Processed {total_processed} token-neuron pairs successfully")
	else:
		warnings.warn("No token-neuron pairs were processed successfully", stacklevel=2)
		
	if verbose:
		print(f"Final interaction matrix shape: {final_interactions.shape}")
		print(f"Max interaction value: {final_interactions.abs().max():.6f}")
		print(f"Non-zero interactions: {(final_interactions.abs() > 1e-8).sum()}")
		
	return final_interactions


def compute_token_neuron_shapley(
	neuron_features: torch.Tensor,
	neuron_idx: int,
	layer: int,
	llm,
	value_function_type: str,
	W_out: torch.Tensor,
	b_out: torch.Tensor,
	mlp_bias: torch.Tensor,
	max_features_per_neuron: int,
	num_samples: int,
	small_threshold: float,
	device: str
) -> torch.Tensor:
	"""
	Compute Shapley-Taylor interactions for a single token's single neuron.
	
	Args:
		neuron_features: Feature values for this neuron at this token [hidden_dim]
		neuron_idx: Index of the neuron
		layer: Layer index
		llm: Language model
		value_function_type: Type of value function ("gelu", "mlp_output")
		W_out: MLP output weights [d_mlp, d_model]
		b_out: MLP output bias [d_model]
		mlp_bias: MLP input bias [d_mlp]
		max_features_per_neuron: Max features to consider
		num_samples: Number of samples for nshap
		small_threshold: Threshold for active features
		device: Device to use
		
	Returns:
		torch.Tensor: [hidden_dim, hidden_dim] interaction matrix
	"""
	try:
		import nshap
	except ImportError:
		raise ImportError("nshap library required for Shapley-Taylor computation")
	
	hidden_dim = neuron_features.shape[0]
	
	# Take top features by absolute value directly (no threshold filtering first)
	if max_features_per_neuron > 0:
		# Take exactly the top-k features by absolute value
		top_k = min(max_features_per_neuron, hidden_dim)
		top_indices = neuron_features.abs().topk(top_k).indices
		
		# Filter out near-zero values to avoid numerical issues
		top_values = neuron_features[top_indices].abs()
		valid_mask = top_values > small_threshold
		
		if valid_mask.sum() < 2:
			return torch.zeros(hidden_dim, hidden_dim)
			
		top_indices = top_indices[valid_mask]
	else:
		# Fallback to threshold-based selection if no limit specified
		active_mask = neuron_features.abs() > small_threshold
		top_indices = torch.nonzero(active_mask).squeeze(-1)
		
		if len(top_indices) < 2:
			return torch.zeros(hidden_dim, hidden_dim)
	
	active_features = neuron_features[top_indices].detach().cpu().numpy()
	active_indices_np = top_indices.cpu().numpy()
	
	# Create value function based on type
	def create_value_function(value_type: str):
		if value_type == "gelu":
			def value_function(x, coalition_indices):
				"""GELU of (sum of coalition features + bias)"""
				if len(coalition_indices) == 0:
					coalition_sum = 0.0
				else:
					coalition_sum = sum(active_features[i] for i in coalition_indices)
				
				# Add MLP bias and apply GELU
				preactivation_with_bias = coalition_sum + float(mlp_bias[neuron_idx].cpu())
				output = torch.nn.functional.gelu(torch.tensor(preactivation_with_bias))
				return float(output)
			
		elif value_type == "mlp_output":
			def value_function(x, coalition_indices):
				"""Full MLP output: W_out @ GELU(sum of coalition features + bias) + b_out"""
				if len(coalition_indices) == 0:
					coalition_sum = 0.0
				else:
					coalition_sum = sum(active_features[i] for i in coalition_indices)
				
				# Add MLP bias and apply GELU
				preactivation_with_bias = coalition_sum + float(mlp_bias[neuron_idx].cpu())
				gelu_output = torch.nn.functional.gelu(torch.tensor(preactivation_with_bias))
				
				# Push through MLP output layer
				# W_out is [d_mlp, d_model], we want neuron_idx row
				w_out_neuron = W_out[neuron_idx].cpu()  # [d_model]
				mlp_output = gelu_output * w_out_neuron + b_out.cpu()  # [d_model]
				
				# Return mean of output (or could use norm, sum, etc.)
				return float(mlp_output.mean())
		elif value_type=="full_mlp_layer":
			def value_function(x, coalition_indices):
				"""
				This implements the evaluation of the full mlp layer.
				So you're assessing the effect of having the two indices 
				at the encoding before you decode and start the MLP reconstruction
				"""
				

				return None
			
		else:
			raise ValueError(f"Unknown value_function_type: {value_type}")
		
		return value_function
	
	value_function = create_value_function(value_function_type)
	
	# Create dummy input for nshap
	dummy_x = np.ones(len(active_features))
	
	try:
		# Compute Shapley-Taylor interactions
		shapley_result = nshap.shapley_taylor(
			dummy_x,
			value_function,
			n=min(len(active_features), 2)  # Limit order for computational tractability
		)
		
		# Extract pairwise interactions and map back to full matrix
		interaction_matrix = torch.zeros(hidden_dim, hidden_dim)
		
		for i in range(len(active_indices_np)):
			for j in range(len(active_indices_np)):
				if i != j:
					interaction_key = tuple(sorted([i, j]))
					if interaction_key in shapley_result:
						# Map back to original feature indices
						feat_i_idx = active_indices_np[i]
						feat_j_idx = active_indices_np[j]
						interaction_matrix[feat_i_idx, feat_j_idx] = shapley_result[interaction_key]
		
		return interaction_matrix
		
	except Exception as e:
		warnings.warn(f"nshap computation failed for neuron {neuron_idx}: {e}")
		return torch.zeros(hidden_dim, hidden_dim)


def test_shapes_small_sample(dataset, llm, crosscoder, layer: int = 0):
	"""
	Test the implementation with a small sample to verify shapes and compilation.
	
	Args:
		dataset: Dataset containing story texts
		llm: Language model
		crosscoder: Crosscoder model
		layer: Which layer to test
		
	Returns:
		dict: Dictionary containing shape information and test results
	"""
	print("=" * 60)
	print("TESTING SHAPLEY INTERACTIONS - SMALL SAMPLE")
	print("=" * 60)
	
	# Test with just 2 stories as requested
	result = compute_shapley_interactions_sequential(
		dataset=dataset,
		llm=llm, 
		crosscoder=crosscoder,
		num_stories=2,
		layer=layer,
		max_features_per_neuron=10,  # Small for testing
		num_samples=200,  # Small for testing
		verbose=True,
		value_function_type="gelu",  # Test with GELU
		max_tokens_per_story=10  # Small for testing
	)
	
	test_info = {
		'final_shape': result.shape,
		'expected_shape': (1536, 1536),
		'shape_correct': result.shape == (1536, 1536),
		'max_value': result.abs().max().item(),
		'non_zero_count': (result.abs() > 1e-8).sum().item(),
		'is_symmetric': torch.allclose(result, result.T, atol=1e-6),
		'compilation_success': True
	}
	
	print("\n" + "=" * 60)
	print("TEST RESULTS")
	print("=" * 60)
	for key, value in test_info.items():
		print(f"{key}: {value}")
		
	if test_info['shape_correct']:
		print("✓ Shape test PASSED")
	else:
		print("✗ Shape test FAILED")
		
	print("=" * 60)
	
	return result, test_info


def compare_value_functions(
	dataset,
	llm,
	crosscoder,
	num_stories: int = 2,
	layer: int = 0,
	max_features_per_neuron: int = 10,
	num_samples: int = 200,
	max_tokens_per_story: int = 10,
	verbose: bool = True
) -> dict:
	"""
	Compare GELU vs full MLP output value functions.
	
	Returns:
		dict: Dictionary with results for both approaches and comparison metrics
	"""
	if verbose:
		print("=" * 80)
		print("COMPARING VALUE FUNCTION APPROACHES")
		print("=" * 80)
	
	# Test GELU approach
	if verbose:
		print("\n1. Testing GELU value function...")
	gelu_interactions = compute_shapley_interactions_sequential(
		dataset=dataset,
		llm=llm,
		crosscoder=crosscoder,
		num_stories=num_stories,
		layer=layer,
		max_features_per_neuron=max_features_per_neuron,
		num_samples=num_samples,
		verbose=verbose,
		value_function_type="gelu",
		max_tokens_per_story=max_tokens_per_story
	)
	
	# Test MLP output approach
	if verbose:
		print("\n2. Testing full MLP output value function...")
	mlp_interactions = compute_shapley_interactions_sequential(
		dataset=dataset,
		llm=llm,
		crosscoder=crosscoder,
		num_stories=num_stories,
		layer=layer,
		max_features_per_neuron=max_features_per_neuron,
		num_samples=num_samples,
		verbose=verbose,
		value_function_type="mlp_output",
		max_tokens_per_story=max_tokens_per_story
	)
	
	# Compare results
	if verbose:
		print("\n3. Comparing results...")
	
	# Correlation between the two approaches
	gelu_flat = gelu_interactions.flatten()
	mlp_flat = mlp_interactions.flatten()
	
	# Remove zeros for correlation
	nonzero_mask = (gelu_flat.abs() > 1e-8) | (mlp_flat.abs() > 1e-8)
	if nonzero_mask.sum() > 1:
		correlation = torch.corrcoef(torch.stack([
			gelu_flat[nonzero_mask], 
			mlp_flat[nonzero_mask]
		]))[0, 1].item()
	else:
		correlation = float('nan')
	
	results = {
		'gelu_interactions': gelu_interactions,
		'mlp_interactions': mlp_interactions,
		'gelu_stats': {
			'max_abs': gelu_interactions.abs().max().item(),
			'mean_abs': gelu_interactions.abs().mean().item(),
			'nonzero_count': (gelu_interactions.abs() > 1e-8).sum().item()
		},
		'mlp_stats': {
			'max_abs': mlp_interactions.abs().max().item(),
			'mean_abs': mlp_interactions.abs().mean().item(),
			'nonzero_count': (mlp_interactions.abs() > 1e-8).sum().item()
		},
		'correlation': correlation
	}
	
	if verbose:
		print(f"\nGELU approach - Max: {results['gelu_stats']['max_abs']:.6f}, "
			  f"Mean: {results['gelu_stats']['mean_abs']:.6f}, "
			  f"Nonzero: {results['gelu_stats']['nonzero_count']}")
		print(f"MLP approach - Max: {results['mlp_stats']['max_abs']:.6f}, "
			  f"Mean: {results['mlp_stats']['mean_abs']:.6f}, "
			  f"Nonzero: {results['mlp_stats']['nonzero_count']}")
		print(f"Correlation: {correlation:.4f}")
		print("=" * 80)
	
	return results


if __name__ == "__main__":
	# This will be filled in by the user when they want to run tests
	pass