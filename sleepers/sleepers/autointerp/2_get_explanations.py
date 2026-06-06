# %%
# %load_ext autoreload
# %autoreload 2

# %%
from sleepers.autointerp.util.llm_autointerp import azureAutointerp
from sleepers.autointerp.util.data_formatting_util import format_activations
import pickle
import os
import pandas as pd
from openai import BadRequestError
from tqdm import tqdm

# %%
# define crosscoder name and activations path

import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Generate explanations for feature activations')
parser.add_argument('--crosscoder_name', type=str, 
                    help='Name of the crosscoder model to analyze')


args = parser.parse_args()
crosscoder_name = args.crosscoder_name
activations_path = f"autointerp_data/collected_activation_data/CC-{crosscoder_name}_10000-samples_withhate.pkl"

explainer = azureAutointerp()
# %%
print(f'trying to load')
with open(activations_path, 'rb') as f:
    activations_data = pickle.load(f)
print(f'loaded')
all_sequences = activations_data['sequences']
activations_data = activations_data['activations']

# %%
explanations = {}
# for all features with non zero activations, get an explanation and save it to a file
for ft_id in tqdm(activations_data.keys()):
    formatted_activations = format_activations(ft_id, all_sequences, activations_data, n_quantiles=0)
    if formatted_activations is None:
        #print(f"Skipping ft_id {ft_id}: No non-zero activations found.")
        continue
    # for example_type in formatted_activations.keys():
    #     for example in formatted_activations[example_type]:
    #         print(example)

    prompt = explainer.format_explainer_prompt(formatted_activations)

    try:
        explanation = explainer.generate_autointerp(prompt)
        #print(f"Ft. {ft_id}: {explanation}")

        if explanation and 'EXPLANATION:' in explanation and "Error - API returned no content" not in explanation:
            try:
                explanations[ft_id] = explanation.split('EXPLANATION:', 1)[1].strip()
                #print(explanations[ft_id])
            except IndexError:
                print(f"Warning: Could not parse explanation for ft_id {ft_id}: '{explanation}'")
                explanations[ft_id] = "Error: Malformed explanation"
        else:
            # Store the error message or a generic error if it's not in the expected format
            error_message = explanation if explanation else "Error: Empty response from API"

    except Exception as e:
        # Check if it's specifically a content filter error on the prompt
        if e.status_code == 400 and e.body and e.body.get('code') == 'content_filter':
            print(f"Azure OpenAI blocked the prompt for ft_id {ft_id} due to content filtering.")
            print("Prompt details (containing triggering examples):")
            # Print the user message content which contains the examples
            user_message_content = next((msg['content'] for msg in prompt if msg['role'] == 'user'), "User message not found in prompt")
            print(user_message_content)
            print(f"Skipping explanation generation for ft_id {ft_id}.\n")
            continue # Skip to the next ft_id
        else:
            print(f"\n--- BadRequestError for ft_id {ft_id} ---")
            print(f"An unexpected BadRequestError occurred: {e}")
            continue # Or just skip this feature

# %%
# save explanations to a csv file
# Create a DataFrame from the explanations dictionary
explanations_df = pd.DataFrame(list(explanations.items()), columns=['feature_id', 'explanation'])

# (Optionally) if file exists, add to it
# if os.path.exists('explanations.csv'):
#     existing_df = pd.read_csv('explanations.csv')
#     explanations_df = pd.concat([existing_df, explanations_df], ignore_index=True)
# Save the DataFrame to a CSV file

explanations_df.to_csv(f'autointerp_data/explanations_{crosscoder_name}_nohate.csv', index=False)

# %%
del activations_data, all_sequences, formatted_activations, prompt, explanation, explanations_df

# %%
