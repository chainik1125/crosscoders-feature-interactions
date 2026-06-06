# Utility functions for formatting data from loaded activations

# Loads activations
# For a given feature, returns optionally any set of: n1 examples of top activations, n2 examples of bottom activations, n3 random examples, n4 examples from each of 25, 50, 75 quantiles of activations
# Each example has k tokens either side of the activating token, as is formatted with the token surrounded by << >>
# Returns a dict of the form:
# {
#     'top_activations': [
#         {
#             'sequence': str with context and <<token>> formatted,
#             'activation': float
#         },
#         ...
#     ],
#     'bottom_activations': [...],
#     'random_activations': [...],
#     '25_quantile_activations': [...],
#     '50_quantile_activations': [...],
#     '75_quantile_activations': [...],
# }

import pickle
import random
import math
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from tqdm import tqdm # Optional: for progress if processing many activations

# Assume these are loaded elsewhere and passed to format_activations
# all_sequences: List[List[str]] = []
# activations_data: Dict[int, Dict[int, List[float]]] = {}

def _format_context_string(
    sequence_tokens: List[str],
    token_index: int,
    context_window: int
) -> str:
    """Helper function to create the formatted context string."""
    start_index = max(0, token_index - context_window)
    end_index = min(len(sequence_tokens), token_index + context_window + 1)

    context_parts = []
    for i in range(start_index, end_index):
        token = sequence_tokens[i]
        if i == token_index:
            context_parts.append(f"<<{token}>>")
        else:
            context_parts.append(token)

    return " ".join(context_parts)

def check_features_for_non_zeros(
    activations_data: Dict[int, Dict[int, List[float]]],
    all_sequences: List[List[str]],
    epsilon: float = 1e-9
):
    """
    Iterates through all features in the activations data and counts how many
    have at least one activation value with abs(value) > epsilon.

    Args:
        activations_data: The loaded activation data dictionary.
        all_sequences: The list of all sequences (used for safety checks).
        epsilon: The threshold to consider an activation non-zero.
    """
    if not activations_data:
        print("CHECK FEATURES: Activations data is empty. Cannot perform check.")
        return

    num_features_total = len(activations_data)
    non_zero_features_count = 0
    max_overall_activation = 0.0
    feature_with_max_act = -1

    print(f"\nCHECK FEATURES: Checking {num_features_total} features for non-zero activations (epsilon={epsilon})...")

    for feature_id in tqdm(activations_data.keys(), desc="Checking features"):
        sequences_for_feature = activations_data.get(feature_id, {})
        found_non_zero_for_feature = False
        for sequence_id, activation_list in sequences_for_feature.items():
            if sequence_id >= len(all_sequences):
                continue # Skip invalid sequence IDs

            for activation_value in activation_list:
                abs_act = abs(activation_value)
                if abs_act > max_overall_activation:
                     max_overall_activation = abs_act
                     feature_with_max_act = feature_id

                if abs_act > epsilon:
                    non_zero_features_count += 1
                    found_non_zero_for_feature = True
                    break # Found one non-zero for this feature, move to the next feature
            if found_non_zero_for_feature:
                break # Move to the next feature

    print(f"CHECK FEATURES: Found {non_zero_features_count} out of {num_features_total} features with at least one non-zero activation.")
    print(f"CHECK FEATURES: Maximum absolute activation value found across all data: {max_overall_activation} (in feature {feature_with_max_act})")

def format_activations(
    feature_id: int,
    all_sequences: List[List[str]],
    activations_data: Dict[int, Dict[int, List[float]]],
    n_top: int = 10,
    n_bottom: int = 15,
    n_random: int = 0,
    n_quantiles: int = 5, # Number of examples per quantile point
    context_window: int = 5,
    epsilon: float = 1e-9, # Threshold to consider an activation non-zero
    format_for_eval: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads activations and formats examples for a given feature.

    For a given feature, returns examples based on activation strength:
    top, bottom (most negative/least positive non-zero), random, and examples
    around the 25th, 50th, and 75th percentiles of non-zero activations.

    Args:
        feature_id: The index of the feature to analyze.
        all_sequences: List of all unique token sequences.
        activations_data: Dict mapping feature_id -> sequence_id -> activation list.
        n_top: Number of top (most positive) activation examples.
        n_bottom: Number of bottom examples, these should be random tokens in random sequences.
        n_random: Number of random non-zero activation examples.
        n_quantiles: Number of examples to take around each quantile point (25, 50, 75).
        context_window: Number of tokens to show on either side of the target token.
        epsilon: Minimum absolute value to consider an activation non-zero.
        format_for_eval: Whether to format for evaluation. If False, tuples are returned instead of formatted examples.
    Returns:
        A dict containing formatted activation examples for different categories.
        Example structure:
        {
            'top_activations': [
                {'activation': float, 'context': str, 'sequence_id': int, 'token_index': int}, ...
            ],
            'bottom_activations': [...],
            'random_activations': [...],
            '25_quantile_activations': [...],
            '50_quantile_activations': [...],
            '75_quantile_activations': [...],
        }
    """
    output_dict = defaultdict(list)
    #print(f"DEBUG: Formatting feature {feature_id} with epsilon={epsilon}") # Added Debug

    if feature_id not in activations_data:
        print(f"Warning: Feature ID {feature_id} not found in activations data.")
        return dict(output_dict) # Return empty dict

    sequences_for_feature = activations_data[feature_id]
    #print(f"DEBUG: Found {len(sequences_for_feature)} sequences for feature {feature_id}") # Added Debug

    # --- Gather all non-zero activations for this feature ---
    feature_specific_activations: List[Tuple[float, int, int]] = [] # (activation_value, sequence_id, token_index)
    processed_count = 0 # Added Debug
    passed_filter_count = 0 # Added Debug
    max_abs_val_seen = 0.0 # Added Debug

    # --- START DEBUG LOOP ---
    #print("DEBUG: Starting loop through sequences...") # Added Debug
    seq_loop_entered = False # Added Debug
    tok_loop_entered = False # Added Debug
    for sequence_id, activation_list in sequences_for_feature.items():
        seq_loop_entered = True # Added Debug
        if sequence_id >= len(all_sequences): # Safety check
             print(f"Warning: Sequence ID {sequence_id} out of bounds for feature {feature_id}. Skipping.")
             continue

    
        for token_index, activation_value in enumerate(activation_list):
            tok_loop_entered = True # Added Debug
            processed_count += 1 # Added Debug
            max_abs_val_seen = max(max_abs_val_seen, abs(activation_value)) # Added Debug

        

            # Only consider activations above the epsilon threshold
            if abs(activation_value) > epsilon: # <----- THIS FILTER
                passed_filter_count += 1 # Added Debug
                feature_specific_activations.append(
                    (activation_value, sequence_id, token_index)
                )

    # --- END DEBUG LOOP ---
    # print(f"DEBUG: Sequence loop entered: {seq_loop_entered}") # Added Debug
    # print(f"DEBUG: Token loop entered: {tok_loop_entered}") # Added Debug
    # print(f"DEBUG: Total activations processed: {processed_count}") # Added Debug
    # print(f"DEBUG: Activations passing epsilon filter: {passed_filter_count}") # Added Debug
    # print(f"DEBUG: Max absolute value seen during processing: {max_abs_val_seen}") # Added Debug

    if not feature_specific_activations:
        #print(f"Warning: No non-zero activations found for feature {feature_id} (using epsilon={epsilon}).") # Modified Warning
        return None

    # --- Sort by activation value ---
    # Sorts from most negative to most positive
    feature_specific_activations.sort(key=lambda x: x[0])
    num_activations = len(feature_specific_activations)

    # --- Helper function to format a list of activation tuples ---
    def _format_example_list(activation_tuples):
        if format_for_eval:
            formatted_list = []
            for act_val, seq_id, tok_idx in activation_tuples:
                if seq_id < len(all_sequences): # Double check sequence ID validity
                 context_str = _format_context_string(
                     all_sequences[seq_id], tok_idx, context_window
                 )
                 formatted_list.append({
                     'activation': act_val,
                     'context': context_str,
                     'sequence_id': seq_id,
                     'token_index': tok_idx
                 })
        else:
            formatted_list = [
                (act_val, seq_id, tok_idx)
                for act_val, seq_id, tok_idx in activation_tuples
            ]
        return formatted_list

    # --- Extract Top Activations ---
    if n_top > 0:
        # Take the last n_top elements (most positive)
        top_tuples = feature_specific_activations[-n_top:]
        output_dict['top_activations'] = _format_example_list(top_tuples)

    # --- Extract Bottom Activations ---
    if n_bottom > 0:
        # get n random sequecne ids
        random_sequence_ids = random.sample(range(len(all_sequences)), n_bottom)
        # sample n_bottom random token positions
        random_token_positions = [random.randint(0, len(all_sequences[seq_id]) - 1) for seq_id in random_sequence_ids]
        bottom_tuples = [(0, seq_id, tok_idx) for seq_id, tok_idx in zip(random_sequence_ids, random_token_positions)]
        output_dict['bottom_activations'] = _format_example_list(bottom_tuples)

    # --- Extract Random Activations ---
    if n_random > 0:
        # Take a random sample without replacement
        num_to_sample = min(n_random, num_activations)
        random_tuples = random.sample(feature_specific_activations, num_to_sample)
        output_dict['random_activations'] = _format_example_list(random_tuples)

    # --- Extract Quantile Activations ---
    if n_quantiles > 0 and num_activations > 1: # Need at least 2 points for quantiles
        quantiles = [0.25, 0.50, 0.75]
        quantile_indices = [int(q * (num_activations - 1)) for q in quantiles] # Index for percentile

        for q_idx, q_val in zip(quantile_indices, quantiles):
            # Define the start and end indices for sampling around the quantile index
            # Ensure we don't go out of bounds and get n_quantile samples if possible
            half_window = n_quantiles // 2
            start_idx = max(0, q_idx - half_window)
            end_idx = min(num_activations, start_idx + n_quantiles)
             # Adjust start if end hit the boundary and we don't have enough samples
            start_idx = max(0, end_idx - n_quantiles)

            quantile_tuples = feature_specific_activations[start_idx:end_idx]
            quantile_key = f'{int(q_val*100)}_quantile_activations'
            output_dict[quantile_key] = _format_example_list(quantile_tuples)


    return dict(output_dict)

# --- Example Usage (requires loading data first) ---
if __name__ == '__main__':
    # 1. Load your data (replace with your actual loading logic)
    pickle_file = 'collected_activation_data/CC-1k68kpv5_10000-samples.pkl' # Example path
    print(f"Loading data from {pickle_file}...")
    try:
        with open(pickle_file, 'rb') as f:
            loaded_data = pickle.load(f)
        all_sequences = loaded_data['sequences']
        activations_data = loaded_data['activations']
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pickle_file}")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- Call the new check function ---
    check_features_for_non_zeros(activations_data, all_sequences)

    # --- START DIAGNOSTICS for a single feature ---
    print(f"\n--- Single Feature Diagnostics ---") # Added header
    print(f"Number of features in activations_data: {len(activations_data)}")
    available_feature_ids = list(activations_data.keys())
    if not available_feature_ids:
        print("Error: activations_data dictionary is empty.")
        exit()
    print(f"Available feature IDs start like: {available_feature_ids[:10]}") # Show first 10

    # Choose the first available feature ID for inspection
    #feature_to_inspect = available_feature_ids[0]
    # Example: Choose a potentially more interesting feature ID if known
    feature_to_inspect = 1197 # Or another ID if you suspect it should be active
    print(f"\nInspecting Feature ID: {feature_to_inspect}")

    if feature_to_inspect not in activations_data:
         print(f"Error: Chosen feature ID {feature_to_inspect} not in activations_data keys!")
         # Don't exit, just skip the single feature formatting
    else:
        sequences_for_feature = activations_data[feature_to_inspect]
        print(f"Number of sequences recorded for feature {feature_to_inspect}: {len(sequences_for_feature)}")

        if not sequences_for_feature:
            print(f"Error: No sequence data found for feature {feature_to_inspect}.")
            # Don't exit, just skip
        else:
            # Inspect the first sequence for this feature
            first_sequence_id = next(iter(sequences_for_feature))
            print(f"Inspecting Sequence ID: {first_sequence_id} for feature {feature_to_inspect}")
            if first_sequence_id >= len(all_sequences):
                print(f"Error: Sequence ID {first_sequence_id} is out of bounds for all_sequences list.")
                # Don't exit, just skip
            else:
                first_activation_list = sequences_for_feature[first_sequence_id]
                print(f"Activation list type: {type(first_activation_list)}")
                if isinstance(first_activation_list, list):
                     print(f"Length of activation list: {len(first_activation_list)}")
                     print(f"First 10 activation values: {first_activation_list[:10]}")
                     # Check for any non-zero values using a small epsilon
                     non_zeros = [val for val in first_activation_list if abs(val) > 1e-9]
                     print(f"Number of values with abs > 1e-9 in this list: {len(non_zeros)}")
                     if non_zeros:
                         print(f"Max absolute value in this list: {max(abs(val) for val in non_zeros)}")
                else:
                    print("Error: Activation data is not a list!")
        # --- END DIAGNOSTICS ---

        # 3. Call the formatting function
        import time
        start_time = time.time()
        formatted_output = format_activations(
            feature_id=feature_to_inspect, # Use the inspected feature ID
            all_sequences=all_sequences,
            activations_data=activations_data,
            n_top=3,
            n_bottom=3,
            n_random=3,
            n_quantiles=1, # Get 1 example per quantile
            context_window=10,
            epsilon=1e-9 # Keep default epsilon for now
        )
        end_time = time.time()
        print(f"Time taken to format activations: {end_time - start_time:.2f} seconds")   

        # 4. Print the results
        print(f"\n--- Formatted Activations for Feature {feature_to_inspect} ---")
        for category, examples in formatted_output.items():
            print(f"\n-- {category} --")
            if examples:
                for i, example in enumerate(examples):
                    print(f"  Example {i+1}:")
                    print(f"    Activation: {example['activation']:.4f}")
                    print(f"    Context: '{example['context']}'")
                    # print(f"    (SeqID: {example['sequence_id']}, TokIDX: {example['token_index']})") # Optional debug info
            else:
                print("  (No examples found for this category)")



