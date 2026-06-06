# %%
# %load_ext autoreload
# %autoreload 2

# %%

import pandas as pd
import pickle
import random
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

from sleepers.autointerp.util.llm_autointerp import azureAutointerp
from sleepers.autointerp.util.data_formatting_util import format_activations

def get_sequences_and_scores(feature_id, all_sequences, activations_data, context_window=5):
    # get formatted activations - tuple of activation values, sequence id, token index
    formatted_activations = format_activations(
        feature_id, 
        all_sequences, 
        activations_data, 
        n_top=10, 
        n_bottom=15, 
        n_quantiles=0,
        format_for_eval=False,
        context_window=5
    )

    #print(formatted_activations)
    if formatted_activations is None:
        print(f"Skipping ft_id {feature_id}: No non-zero activations found.")
        return None
    
    # get sequences with top activating tokens
    top_sequence_ids = [seq[1] for seq in formatted_activations['top_activations']]
    top_token_ids = [seq[2] for seq in formatted_activations['top_activations']]
    # get sequences with no activating tokens
    false_sequence_ids = [seq[1] for seq in formatted_activations['bottom_activations']]
    false_token_ids = [seq[2] for seq in formatted_activations['bottom_activations']]
    # get full sequences with top and no activation tokens
    top_sequences = []
    false_sequences = []
    
    for i in range(len(top_sequence_ids)):
        sequence_tokens = all_sequences[top_sequence_ids[i]]
        start_index = max(0, top_token_ids[i] - context_window)
        end_index = min(len(sequence_tokens), top_token_ids[i] + context_window + 1)
        top_sequences.append(sequence_tokens[start_index:end_index])
    for i in range(len(false_sequence_ids)):
        sequence_tokens = all_sequences[false_sequence_ids[i]]
        start_index = max(0, false_token_ids[i] - context_window)
        end_index = min(len(sequence_tokens), false_token_ids[i] + context_window + 1)
        false_sequences.append(sequence_tokens[start_index:end_index])

    combined_sequences = top_sequences + false_sequences
    combined_scores = [1] * len(top_sequences) + [0] * len(false_sequences)
    # shuffle the sequences and scores
    combined_sequences, combined_scores = shuffle(combined_sequences, combined_scores)
    return combined_sequences, combined_scores
    


def get_autointerp_eval_output(evaluator, feature_id, all_sequences, activations_data, explanation_df):
    # get sequences and scores
    combined_sequences, correct_scores = get_sequences_and_scores(feature_id, all_sequences, activations_data)
    # get explanations
    explanations = explanation_df[explanation_df['feature_id'] == feature_id]['explanation'].tolist()
    prompt = evaluator.format_evaluator_prompt(explanations[0], combined_sequences)
    response = evaluator.generate_autointerp(prompt)
    return response, correct_scores

def get_confusion_matrix_stats(scores, correct_scores):
    # get confusion matrix
    scores = np.array(scores)
    correct_scores = np.array(correct_scores)
    try:
        false_positives = np.where((scores == 1) & (correct_scores == 0))[0]
        false_negatives = np.where((scores == 0) & (correct_scores == 1))[0]
        true_positives = np.where((scores == 1) & (correct_scores == 1))[0]
        true_negatives = np.where((scores == 0) & (correct_scores == 0))[0]
    except:
        print(scores)
        print(correct_scores)
        return None, None, None, None
    return false_positives, false_negatives, true_positives, true_negatives


# %%
# load activations
import argparse
base_dir = 'autointerp_data'
# Set up argument parsing
parser = argparse.ArgumentParser(description='Evaluate feature explanations')
parser.add_argument('--crosscoder_name', type=str,
                    help='Name of the crosscoder model to evaluate')
args = parser.parse_args()
crosscoder_name = args.crosscoder_name
#crosscoder_name = "86u64trx"
activations_path = f"{base_dir}/collected_activation_data/CC-{crosscoder_name}_10000-samples_withhate.pkl"


with open(activations_path, 'rb') as f:
    activations_data = pickle.load(f)
all_sequences = activations_data['sequences']
activations_data = activations_data['activations']

# load explanation_df

explanation_df = pd.read_csv(f'{base_dir}/explanations_{crosscoder_name}_withhate.csv')

evaluator = azureAutointerp()

# %%
metrics = []
for feature_id in tqdm(explanation_df['feature_id'].unique()):
    output, correct_scores = get_autointerp_eval_output(evaluator,feature_id, all_sequences, activations_data, explanation_df)
    # get scores from output
    try:
        scores = output.strip().replace('[', '').replace(']', '')
        scores = [int(score) for score in scores.split(',')]
    except:
        print(f'Could not get scores from output for feature {feature_id}, continuing...')
        print('Output:', output)
        continue

    false_positives, false_negatives, true_positives, true_negatives = get_confusion_matrix_stats(scores, correct_scores)
    explanation = explanation_df[explanation_df['feature_id'] == feature_id]['explanation'].tolist()
    if false_positives is None:
        # Error in computing confusion stats; default all counts to zero
        metrics.append({
            'feature_id': feature_id,
            'explanation': explanation,
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'true_negatives': 0
        })
    else:
        metrics.append({
            'feature_id': feature_id,
            'explanation': explanation,
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'true_positives': len(true_positives),
            'true_negatives': len(true_negatives)
        })

# %%
# save to new csv
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f'{base_dir}/autointerp_eval_metrics_{crosscoder_name}_withhate.csv', index=False)

del activations_data, all_sequences, metrics_df

# %%
