# %%
import torch
from tqdm import tqdm
from collections import defaultdict
from typing import Any, List, Dict, Iterable, Tuple
import numpy as np
import os
import pickle
from datasets import load_dataset
import sys

from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.autointerp.util.activation_util import get_activations_batch




def collect_activating_tokens_with_context(
    dataset: Iterable[Dict[str, str]], # Expects dicts with a "text" key
    model: Any,
    crosscoder: Any,
    output_file: str,
    num_samples: int = 1000,
    batch_size: int = 32,
    seq_len: int = 128 # Sequence length used in get_activations_batch
):
    """
    Collects feature activations and stores the full sequences for context retrieval.

    Iterates through the dataset, processes texts in batches to get feature activations,
    and stores the full token sequence along with the full sequence of activation
    strengths for each feature.

    Saves data in a dictionary:
    {
        'sequences': List[List[str]],  # List of unique token sequences [Num_Unique_Seqs, Seq_Len]
        'activations': Dict[int, Dict[int, List[float]]] # {feature_idx: {sequence_id: [act_strength_tok0, act_strength_tok1, ...]}}
    }

    Args:
        dataset: An iterable yielding dictionaries, each containing a "text" key.
        model: The transformer-lens model.
        crosscoder: The crosscoder model.
        output_file: Path to save the collected data (using pickle).
        num_samples: Maximum number of samples to process from the dataset.
        batch_size: Number of samples to process in each batch.
        seq_len: The fixed sequence length for tokenization (must match get_activations_batch).
    """
    device = next(crosscoder.parameters()).device
    num_features = crosscoder.W_dec_HXD.shape[0] # H
    print(f"Using device: {device}. Collecting tokens for {num_features} features.")

    # --- Data Structures ---
    # Stores unique sequences encountered: List[List[str]]
    all_sequences: List[List[str]] = []
    # Maps tuple(sequence_tokens) -> sequence_id for quick lookup
    sequence_map: Dict[Tuple[str, ...], int] = {}
    # Stores activation info: {feature_idx: {sequence_id: [token_idx1, ...]}}
    activating_indices: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    num_processed = 0
    dataset_iterator = iter(dataset)

    with torch.no_grad(): # Ensure no gradients are computed
        for i in tqdm(range(0, num_samples, batch_size), desc="Collecting activating tokens"):
            batch_texts = []
            batch_token_strings_list = [] # Store lists of string tokens for this batch
            batch_sequence_ids = [] # Store sequence IDs for this batch

            # --- Prepare Batch and Manage Sequences ---
            for _ in range(batch_size):
                try:
                    sample = next(dataset_iterator)
                    text = sample["text"]
                    batch_texts.append(text)

                    # Tokenize and get string tokens, padding/truncating
                    tokens_ids = model.tokenizer.encode(text)[:seq_len]
                    tokens_strings = model.tokenizer.batch_decode(
                        [[t] for t in tokens_ids] # Decode one by one
                    )
                    # Pad with a specific PAD token string if needed
                    if len(tokens_strings) < seq_len:
                        tokens_strings.extend(["<PAD>"] * (seq_len - len(tokens_strings)))
                    batch_token_strings_list.append(tokens_strings) # Keep as list for now

                    # --- Manage Unique Sequences ---
                    tokens_tuple = tuple(tokens_strings) # Use tuple for dict key
                    if tokens_tuple not in sequence_map:
                        sequence_id = len(all_sequences)
                        sequence_map[tokens_tuple] = sequence_id
                        all_sequences.append(tokens_strings) # Store the list
                    else:
                        sequence_id = sequence_map[tokens_tuple]
                    batch_sequence_ids.append(sequence_id)
                    # --- End Manage Unique Sequences ---

                    num_processed += 1
                    if num_processed >= num_samples:
                        break
                except StopIteration:
                    break # End of dataset

            if not batch_texts:
                break # No more data

            current_batch_size = len(batch_texts)

            # --- Get Activations ---
            # feature_activations_BSH: [B, S, H]
            feature_activations_BSH, _ = get_activations_batch(batch_texts, model, crosscoder)
            feature_activations_BSH = feature_activations_BSH.cpu() # Move to CPU for processing

            # --- Find and Store Activating Indices ---
            active_mask_BSH = feature_activations_BSH != 0 # [B, S, H]

            for b in range(current_batch_size):
                sequence_id = batch_sequence_ids[b]
                

                # Iterate through ALL features
                for h in range(num_features):
                    acts_current_feature_S = feature_activations_BSH[b, :, h] # Shape [S]
                    activating_indices[h][sequence_id] = acts_current_feature_S.cpu().tolist() # 
            
            del feature_activations_BSH, active_mask_BSH
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Save Results ---
    # Convert defaultdicts back to dicts for saving
    final_activating_indices = {
        feat: dict(seq_map) for feat, seq_map in activating_indices.items()
    }
    output_data = {
        'sequences': all_sequences,
        'activations': final_activating_indices
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Use pickle for potentially complex nested dict/list structure
    print(f"Saving data to {output_file} - this may take a while...")
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Saved activating token indices and sequences for {len(final_activating_indices)} features to {output_file}")
    print(f"Total unique sequences stored: {len(all_sequences)}")
    print(f"Total samples processed: {num_processed}")

# %%
# --- Main function to run the collection
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Configuration for loading models and data ---
    DATASET_NAME = "mars-jason-25/tiny_stories_instruct_sleeper_data"
    LLM_BASE_MODEL_REPO = "roneneldan/TinyStories-Instruct-33M"
    LLM_LORA_MODEL_REPO = "mars-jason-25/tiny-stories-33M-TSdata-ft1" # Fine-tuned on sleeper data
    #WANDB_RUN_NAME_CROSSCODER = "86u64trx"  # Base XC, l=1000
    import argparse
    parser = argparse.ArgumentParser(description='Collect autointerp data')
    parser.add_argument('--crosscoder_name', type=str, default="ckubmeg1",
                    help='Name of the crosscoder model to analyze')
    args = parser.parse_args()

    
    
    WANDB_RUN_NAME_CROSSCODER = args.crosscoder_name  # Penalised XC, l=1000
    WANDB_ENTITY = "dmitry2-uiuc"
    WANDB_PROJECT = "sleeper-model-diffing"
    WANDB_ARTIFACTS_PATH = "../../.wandb_artifacts"
    
    CACHE_DIR = "./.cache" 
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- Load Dataset ---
    print(f"Loading dataset: {DATASET_NAME}")
    # Load a small subset for demonstration
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)

    
    
    
    #filter to exclude I hate you's.
    dataset = dataset.filter(lambda x: x['is_training'] == True)
    
    
    
    print(f"Dataset loaded. Number of samples: {len(dataset)}")
    print(f"Text samples:")
    for i in range(5):
        print(dataset[i]["text"])
    
    


    # --- Load LLM ---
    print(f"Loading LLM: {LLM_BASE_MODEL_REPO} with LoRA: {LLM_LORA_MODEL_REPO}")
    llm = build_llm_lora(
        base_model_repo=LLM_BASE_MODEL_REPO,
        lora_model_repo=LLM_LORA_MODEL_REPO,
        cache_dir=CACHE_DIR,
        device=DEVICE,
        dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32, # Use float16 on GPU
    )
    print("LLM loaded.")

    # --- Load Crosscoder ---
    print(f"Loading Crosscoder from WandB run: {WANDB_RUN_NAME_CROSSCODER}")
    crosscoder = load_crosscoder_from_wandb(
        WANDB_ENTITY, WANDB_PROJECT, WANDB_RUN_NAME_CROSSCODER, WANDB_ARTIFACTS_PATH, DEVICE
    )
    crosscoder.to(DEVICE) # Ensure crosscoder is on the correct device
    print("Crosscoder loaded.")

    num_samples_to_collect = 10_000

    # --- Configuration for data collection ---
    output_dir = "autointerp_data/collected_activation_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir,f"CC-{WANDB_RUN_NAME_CROSSCODER}_{num_samples_to_collect}-samples_nohate.pkl")

    batch_size_collection = 10 # Adjust based on your GPU memory
    # This seq_len should ideally match the one used when training/evaluating the crosscoder
    # I think this is right??
    sequence_length = 128 

    print(f"\nStarting data collection for {num_samples_to_collect} samples...")
    collect_activating_tokens_with_context( 
        dataset=dataset,
        model=llm,
        crosscoder=crosscoder,
        output_file=output_file_path,
        num_samples=num_samples_to_collect,
        batch_size=batch_size_collection,
        seq_len=sequence_length,
    )
    print(f"Data collection finished. Saved to {output_file_path}")

if __name__ == "__main__":
    main()

# %%
