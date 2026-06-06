# %%
%load_ext autoreload
%autoreload 2
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from typing import Any, List, Dict, Iterable, Tuple
import numpy as np
import os
import pickle
from datasets import load_dataset
import wandb
#import seaborn as sns
import matplotlib.pyplot as plt

from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.autointerp.util.activation_util import get_activations_batch
from sleepers.analysis.analysis_utils import feature_interactions_sum

# %%


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Configuration for loading models and data ---
DATASET_NAME = "mars-jason-25/tiny_stories_instruct_sleeper_data"
LLM_BASE_MODEL_REPO = "roneneldan/TinyStories-Instruct-33M"
LLM_LORA_MODEL_REPO = "mars-jason-25/tiny-stories-33M-TSdata-ft1" # Fine-tuned on sleeper data
WANDB_RUN_NAME_CROSSCODER = "86u64trx"  # Base XC, l=1000
WANDB_RUN_NAME_CROSSCODER = "ckubmeg1"  # Penalised XC, l=1000
WANDB_ENTITY = "dmitry2-uiuc"
WANDB_PROJECT = "sleeper-model-diffing"
WANDB_ARTIFACTS_PATH = "../../.wandb_artifacts"

CACHE_DIR = "./.cache" 
os.makedirs(CACHE_DIR, exist_ok=True)

# # --- Load Dataset ---
# print(f"Loading dataset: {DATASET_NAME}")
# # Load a small subset for demonstration
# dataset = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)
# print(f"Dataset loaded. Number of samples: {len(dataset)}")

# --- Load LLM ---
# ''print(f"Loading LLM: {LLM_BASE_MODEL_REPO} with LoRA: {LLM_LORA_MODEL_REPO}")
# llm = build_llm_lora(
#     base_model_repo=LLM_BASE_MODEL_REPO,
#     lora_model_repo=LLM_LORA_MODEL_REPO,
#     cache_dir=CACHE_DIR,
#     device=DEVICE,
#     dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32, # Use float16 on GPU
# )
# print("LLM loaded.")''

# --- Load Crosscoder ---
print(f"Loading Crosscoder from WandB run: {WANDB_RUN_NAME_CROSSCODER}")
crosscoder = load_crosscoder_from_wandb(
    WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME_CROSSCODER, WANDB_ARTIFACTS_PATH, DEVICE
)
crosscoder.to(DEVICE) # Ensure crosscoder is on the correct device
print("Crosscoder loaded.")


# %%
# Inputs: layer, num_datapoints,dataset,llm,crosscoder, num_features=1536
# Outputs: feature_interactions = (num_features, num_features)

# get cosine sim matrix based on the crosscoder decoder
features = crosscoder.W_dec_HXD.reshape(crosscoder.W_dec_HXD.shape[0], -1).cpu()  # Move to CPU
print(features.shape)

# Calculate cosine similarity between all feature pairs in batches to save memory
batch_size = 128  # Adjust based on your memory constraints
num_features = features.shape[0]
cosine_sim_matrix = torch.zeros((num_features, num_features), device=torch.device('cpu'))

for i in tqdm(range(0, num_features, batch_size)):
    # Process in batches
    i_end = min(i + batch_size, num_features)
    features_i = features[i:i_end]
    
    for j in range(0, num_features, batch_size):
        j_end = min(j + batch_size, num_features)
        features_j = features[j:j_end]
        
        # Compute similarities for this batch
        # Normalize the features for cosine similarity calculation
        features_i_norm = F.normalize(features_i, p=2, dim=1)
        features_j_norm = F.normalize(features_j, p=2, dim=1)
        
        # Compute batch of similarities at once
        batch_sim = torch.mm(features_i_norm, features_j_norm.t())
        
        # Store in the result matrix
        cosine_sim_matrix[i:i_end, j:j_end] = batch_sim

print(f"Cosine similarity matrix shape: {cosine_sim_matrix.shape}")  # Should be (1536, 1536)

# %%
# load summed_feat_ints_ckubmeg1.pkl
with open(f'summed_feat_ints_{WANDB_RUN_NAME_CROSSCODER}.pkl', 'rb') as f:
    summed_feat_ints = pickle.load(f)
print(summed_feat_ints.shape)

# sum across first dimension
summed_feat_ints_fm = summed_feat_ints.sum(dim=0)
print(summed_feat_ints_fm.shape)


# %%

import pandas as pd
# load the explaantions file
explanations_file = '/workspace/crosscoders-feature-interactions/sleepers/sleepers/autointerp/autointerp_data/explanations_ckubmeg1.csv'
explanations = pd.read_csv(explanations_file)
print(explanations.shape)
# get list of all feature ids
feature_ids = explanations['feature_id'].unique()
# change to list of ints
feature_ids = list(feature_ids)
feature_ids = [int(id) for id in feature_ids]
print(len(feature_ids))


# remove all others from the summed_feat_ints_fm
summed_feat_ints_active = summed_feat_ints_fm[feature_ids, :][:, feature_ids]
print(summed_feat_ints_active.shape)

top_10_values, top_10_indices = torch.topk(summed_feat_ints_active.view(-1), k=100)
print(top_10_values)
print(top_10_indices)
top_10_indices_2d = torch.stack([top_10_indices // len(feature_ids), top_10_indices % len(feature_ids)], dim=1)
print(top_10_indices_2d)

# %%
# get the top 10 explanations                                                       
for id1, id2 in top_10_indices_2d:
    # get correct index in explanations dataframe
    id1_idx = feature_ids[id1]
    id2_idx = feature_ids[id2]
    print(id1_idx, id2_idx)
    # check if feature id1 is in the explanations dataframe
    if id1_idx in explanations['feature_id'].values:
        print(explanations[explanations['feature_id'] == id1_idx]['explanation'].values[0])
    else:
        print(f"Feature {id1} not found in explanations")
    if id2_idx in explanations['feature_id'].values:
        print(explanations[explanations['feature_id'] == id2_idx]['explanation'].values[0])
    else:
        print(f"Feature {id2} not found in explanations")
    print("--------------------------------")
# %%

# do same for cosine similarity matrix
cosine_sim_matrix_active = cosine_sim_matrix[feature_ids, :][:, feature_ids]
# set bottom triangle to 0
cosine_sim_matrix_active.tril_(0)
# set diagonal to 0
cosine_sim_matrix_active.fill_diagonal_(0)
print(cosine_sim_matrix_active.shape)

top_10_values, top_10_indices = torch.topk(cosine_sim_matrix_active.view(-1), k=10)
print(top_10_values)
print(top_10_indices)
top_10_indices_2d_cosine = torch.stack([top_10_indices // len(feature_ids), top_10_indices % len(feature_ids)], dim=1)
print(top_10_indices_2d_cosine)

# %%
# get the top 10 explanations
for id1, id2 in top_10_indices_2d_cosine:
    id1_idx = feature_ids[id1]
    id2_idx = feature_ids[id2]
    print(id1_idx, id2_idx)
    if id1_idx in explanations['feature_id'].values:
        print(explanations[explanations['feature_id'] == id1_idx]['explanation'].values[0])
    else:
        print(f"Feature {id1} not found in explanations")
    if id2_idx in explanations['feature_id'].values:
        print(explanations[explanations['feature_id'] == id2_idx]['explanation'].values[0])
    else:
        print(f"Feature {id2} not found in explanations")
    print("--------------------------------")
# %%

# extract examples of top activations from the dataset for each feature
# load activations data