import numpy as np
from matplotlib import pyplot as plt
import sys
import torch
from pathlib import Path
from einops import einsum
from sleepers.scripts.utils import load_crosscoder_from_wandb
import wandb
from datasets import load_dataset




wandb_run_name = 'h2mwu2g7'



# load crosscoder decoder features
crosscoder = load_crosscoder_from_wandb(
    "dmitry2-uiuc",
    "sleeper-model-diffing",
    wandb_run_name,
    "../../.wandb_artifacts",
    DEVICE)

hookpoints = [
    "blocks.0.hook_resid_pre",
    "blocks.0.ln1.hook_normalized",
    "blocks.0.hook_resid_mid",
    "blocks.0.ln2.hook_normalized",
    "blocks.1.hook_resid_pre",
    "blocks.1.ln1.hook_normalized",
    "blocks.1.hook_resid_mid",
    "blocks.1.ln2.hook_normalized",
    "blocks.2.hook_resid_pre",
    "blocks.2.ln1.hook_normalized",
    "blocks.2.hook_resid_mid",
    "blocks.2.ln2.hook_normalized",
    "blocks.3.hook_resid_pre",
    "blocks.3.ln1.hook_normalized",
    "blocks.3.hook_resid_mid",
    "blocks.3.ln2.hook_normalized",
    "blocks.3.hook_resid_post",
]


api = wandb.Api()
artifact = api.artifact(f"dmitry2-uiuc/sleeper-model-diffing/dataloader-means_run-{wandb_run_name}:latest")
artifact_dir = Path(artifact.download(root="../../.wandb_artifacts"))
dataloader_mean_SMPD = torch.load(artifact_dir / "dataloader_means.pt", map_location=DEVICE)



dataset = load_dataset('mars-jason-25/tiny_stories_instruct_sleeper_data', split='train')
dataset = dataset.filter(lambda x: x['is_training'] == True)

from sleepers.scripts.llms import build_llm_lora
llm = build_llm_lora(
    base_model_repo="roneneldan/TinyStories-Instruct-33M",
    lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
    cache_dir=None,
    device=DEVICE,
    dtype=None
)
tokenizer = llm.tokenizer



def attention_pattern_QK(layer, head, q_input_QD, q_do_bias, k_input_KD, k_do_bias, model=llm):
    W_Q = model.blocks[layer].attn.W_Q[head]
    b_Q = model.blocks[layer].attn.b_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    b_K = model.blocks[layer].attn.b_K[head]
    q = einsum(W_Q, q_input_QD, "d a, s d -> s a")
    if q_do_bias:
        q += b_Q
    k = einsum(W_K, k_input_KD, "d a, s d -> s a")
    if k_do_bias:
        k += b_K
    attention_scores = einsum(q, k, "q a, k a -> q k")
    return attention_scores.to("cpu").numpy()

def lower_triangular_mask(pattern):
    mask = np.triu(np.ones(pattern.shape), k=1)
    return np.ma.array(np.tril(pattern, k=0), mask=mask)

def softmax_pattern(attention_pattern):
    attention_pattern = np.tril(attention_pattern)
    softmaxed_attn_scores = np.zeros_like(attention_pattern)
    for q in range(0, 128):
        attn = attention_pattern[q,:q+1]
        attn_max = max(attn)
        attn_exp = np.exp(attn-attn_max)
        attn_softmax = attn_exp/attn_exp.sum()
        softmaxed_attn_scores[q,:q+1] = attn_softmax
    return lower_triangular_mask(softmaxed_attn_scores)

def pattern_subtract_row_mean(pattern):
    mean_subtracted_pattern = np.zeros_like(pattern)
    for q in range(pattern.shape[0]):
        row_mean = pattern[q,:q+1].mean()
        mean_subtracted_pattern[q,:q+1] = pattern[q,:q+1]-row_mean
    return mean_subtracted_pattern


def main():
    print(sys.executable)
    checkpoint_dir = Path('../../.checkpoints/')
    print(torch.__version__)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)

    torch.set_grad_enabled(False)

    


if __name__ == "__main__":
    print(f'the main character')
