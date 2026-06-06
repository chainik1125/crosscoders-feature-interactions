from datetime import datetime
from pathlib import Path
import wandb
import torch
import einops
from model_diffing.models.crosscoder import AcausalCrosscoder
from typing import Any

def neuron_sharpness_quick(enc_acts_BH:torch.Tensor,crosscoder:object,W_ins:torch.Tensor,b_ins:torch.Tensor,W_outs:torch.Tensor,b_outs:torch.Tensor,device:str="cpu"):
    # W_ins=torch.stack([llm.blocks[val].mlp.W_in for val in range(4)],dim=0).to(DEVICE)
    # b_ins=torch.stack([llm.blocks[val].mlp.b_in for val in range(4)],dim=0).to(DEVICE)	
    # W_outs=torch.stack([llm.blocks[val].mlp.W_out for val in range(4)],dim=0).to(DEVICE)	
    # b_outs=torch.stack([llm.blocks[val].mlp.b_out for val in range(4)],dim=0).to(DEVICE)

    #crosscoder=crosscoder.to(DEVICE)
    
    # Get the number of blocks dynamically from W_ins shape
    n_blocks = W_ins.shape[0]
    
    # Calculate hookpoints per block (assuming 4 hookpoints per layer for consistency)
    hookpoints_per_block = crosscoder.W_dec_HXD.shape[2] // n_blocks
    
    W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,hookpoints_per_block*block + (hookpoints_per_block-1),:] for block in range(n_blocks)],dim=0)#.to(DEVICE)
    b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,hookpoints_per_block*block + (hookpoints_per_block-1),:] for block in range(n_blocks)],dim=0)#.to(DEVICE)
    
    
    enc_weights=enc_acts_BH[enc_acts_BH != 0].mean(dim=0)
    data_ind_w=einops.einsum(W_ins,W_dec_PHD,"block d_model d_mlp, block hidden d_model -> block d_mlp hidden",)
    data_ind_b=b_ins[:,:,None]/W_dec_PHD.shape[1]
    data_weighted_w=enc_weights.unsqueeze(0).unsqueeze(0)*data_ind_w
    #p_PNH=data_ind_w+0*data_ind_b
    p_PNH=data_weighted_w+0*data_ind_b
    
    return p_PNH


def calculate_fvu_X(
    y_BXD: torch.Tensor,
    y_pred_BXD: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    For each crosscoding output space (model, hookpoint, token, etc.) Calculates the fvu, returning
    a tensor of shape (crosscoding_dim_1, crosscoding_dim_2, ...) where each element is the fvu for the corresponding
    crosscoding output space.

    see https://www.lesswrong.com/posts/ZBjhp6zwfE8o8yfni/#Rm8xDeB95fb2usorb for a discussion of this
    """
    y_mean_BXD = y_BXD.mean(dim=0, keepdim=True)

    var_err_BX = (y_BXD - y_pred_BXD).norm(p=2, dim=-1).square()  # variance 
    var_err_X = var_err_BX.mean(0)  # mean over batch

    var_total_BX = (y_BXD - y_mean_BXD).norm(p=2, dim=-1).square()
    var_total_X = var_total_BX.mean(0)  # mean over batch

    return var_err_X / (var_total_X + eps)


def sharpness_func(acts_BXH:torch.Tensor,beta:float=1.0,llm:Any=None,eps:float=1e-8) -> float:
    """
    Calculate the swish sharpness for a given tensor of activations.
    I want this to work for all of: the encoding (shape BH), the decoding (shape BSMLD),
    and the push-through to the activations (shape BSMLHN)
    """
    
    neuron_sharpness = (acts_BXH.abs().sum(dim=-1)+eps)/((acts_BXH.abs()*torch.softmax(beta*acts_BXH.abs(),dim=-1)).sum(dim=-1)+eps)-1

    #for reference, calculate the mean ratio:
    #mean_max_ratio=torch.max(acts_BXH,dim=-1)[0]/acts_BXH.abs().sum(dim=-1)
    #mean_max_ratio=mean_max_ratio.mean()
    
    #TODO: add the other two
    # elif feature_func=="max_ratio":
    #     neuron_sharpness = preacts_BNH.abs().sum(dim=-1)/(preacts_BNH.abs().max(dim=-1)[0])-1
    # elif feature_func=="gini":
    #     preacts_vals,preacts_idx=torch.sort(preacts_BNH.abs(),dim=-1,descending=False)
    #     features=preacts_BNH.shape[-1]
    #     neuron_sharpness=(((2*torch.arange(1,features+1)-features-1))*preacts_vals).sum(dim=-1)/(features*preacts_vals.sum(dim=-1))
    # else:
    #     neuron_sharpness = feature_func(preacts_BNH)
    
    #now average over everything - batch, sequence, model, layer
    neuron_sharpness=neuron_sharpness.mean()
    
    return neuron_sharpness

def get_neuron_preacts(enc_acts_BH:torch.Tensor,W_dec_PHD:torch.Tensor,b_dec_PD:torch.Tensor,W_ins:torch.Tensor,b_ins:torch.Tensor,W_outs:torch.Tensor,b_outs:torch.Tensor,device:str="cpu",bias:bool=True):
    
    # print(f'W_ins.shape: {W_ins.shape}')
    # print(f'b_ins.shape: {b_ins.shape}')
    # print(f'W_outs.shape: {W_outs.shape}')
    # print(f'b_outs.shape: {b_outs.shape}')
        
    # W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in range(4)],dim=0)
    # b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
    
    hidden_dim=W_dec_PHD.shape[1]
    enc_acts_BH=enc_acts_BH.to(device)
    W_dec_PHD=W_dec_PHD.to(device)
    b_dec_PD=b_dec_PD.to(device)
    W_ins=W_ins.to(device)
    b_ins=b_ins.to(device)
    W_outs=W_outs.to(device)
    b_outs=b_outs.to(device)

    enc_BHD_W = einops.einsum(enc_acts_BH[...,None], W_dec_PHD[None,:,:], "batch hidden one, one block hidden d_model -> block batch d_model hidden")
    #print(f'enc_BHD_W.shape: {enc_BHD_W.shape}')
    enc_BHD_b = bias*b_dec_PD[:,None,:,None]/hidden_dim
    #print(f'enc_BHD_b.shape: {enc_BHD_b.shape}')
    p_BNH = einops.einsum(W_ins, enc_BHD_W+enc_BHD_b, "block d_model d_mlp, block batch d_model hidden -> block batch d_mlp hidden")
    #print(f'p_BNH.shape: {p_BNH.shape}')
    #print(f'b_ins.shape: {b_ins.shape}')
    
    p_BNH += bias*b_ins[:,None,:,None]/hidden_dim
    #print(f'p_BNH.shape: {p_BNH.shape}')
    
    return p_BNH

def get_neuron_preacts_cutoff(enc_acts_BH:torch.Tensor,W_dec_PHD:torch.Tensor,b_dec_PD:torch.Tensor,W_ins:torch.Tensor,b_ins:torch.Tensor,W_outs:torch.Tensor,b_outs:torch.Tensor,device:str="cpu",bias:float=0,block_idx:int=None,precomputed_sort=None):
    """
    Idea of this calculation is to cutoff the encoding past the point
    where the contributions are negligible.
    
    Args:
        block_idx: If specified, process only this block and return (batch, d_mlp, hidden).
                  If None, process all blocks and return (blocks, batch, d_mlp, hidden).
    """

    # If processing a single block for memory efficiency
    if block_idx is not None:
        # Extract weights for just this block
        W_dec_HD = W_dec_PHD[block_idx]  # (hidden, d_model)
        b_dec_D = b_dec_PD[block_idx]    # (d_model,)
        W_in = W_ins[block_idx]          # (d_model, d_mlp)  
        b_in = b_ins[block_idx]          # (d_mlp,)
        
        hidden_dim = W_dec_HD.shape[0]
        
        # MAJOR OPTIMIZATION: Use precomputed sorting if available (avoids 12x redundant sorts)
        if precomputed_sort is not None:
            sorted_enc_vals, sorted_enc_inds, max_non_zero_index = precomputed_sort
        else:
            # Sort activations by absolute value (keep original behavior)
            sorted_enc_vals, sorted_enc_inds = torch.sort(torch.abs(enc_acts_BH), dim=-1, descending=True)
            
            # Find max non-zero index (keep original logic)
            non_zero_indices = (sorted_enc_vals != 0).sum(dim=1)
            max_non_zero_index = non_zero_indices.max().item()
        
        # Early exit if no non-zero activations
        if max_non_zero_index == 0:
            batch_size = enc_acts_BH.shape[0]
            p_BNH_single = torch.zeros(batch_size, W_in.shape[1], 1, device=enc_acts_BH.device)
            return p_BNH_single, sorted_enc_inds[:, :1]
        
        # Filter to non-zero activations
        filtered_sorted_enc_BH = sorted_enc_vals[:, :max_non_zero_index]
        filtered_sorted_inds = sorted_enc_inds[:, :max_non_zero_index]
        
        # OPTIMIZATION: Use gather for indexing (more efficient than advanced indexing)
        # Original: W_dec_PBHD = W_dec_HD[sorted_enc_inds[:, :max_non_zero_index], :]
        batch_size = enc_acts_BH.shape[0]
        expanded_inds = filtered_sorted_inds.unsqueeze(-1).expand(-1, -1, W_dec_HD.shape[1])
        W_dec_BHD = torch.gather(W_dec_HD.unsqueeze(0).expand(batch_size, -1, -1), 1, expanded_inds)
        
        # OPTIMIZATION: Use einops but with contiguous tensors for better performance
        # First einsum: maintain original shape logic
        enc_BHD_W = einops.einsum(filtered_sorted_enc_BH[..., None], W_dec_BHD, 
                                 "batch hidden_c one, batch hidden_c d_model -> batch d_model hidden_c")
        enc_BHD_b = bias * b_dec_D[None, :, None] / hidden_dim
        
        # Second einsum: push through MLP - maintain original shape  
        p_BNH_single = einops.einsum(W_in, enc_BHD_W + enc_BHD_b, 
                                    "d_model d_mlp, batch d_model hidden -> batch d_mlp hidden")
        p_BNH_single += bias * b_in[None, :, None] / hidden_dim
        
        # FINAL OPTIMIZATION: Make output contiguous for faster downstream operations
        p_BNH_single = p_BNH_single.contiguous()
        
        return p_BNH_single, sorted_enc_inds
    
    # Original behavior for backward compatibility
    hidden_dim=W_dec_PHD.shape[1]
    #enc_acts_BH=enc_acts_BH.to(device)
    #W_dec_PHD=W_dec_PHD.to(device)
    #b_dec_PD=b_dec_PD.to(device)
    #W_ins=W_ins.to(device)
    #b_ins=b_ins.to(device)
    #W_outs=W_outs.to(device)
    #b_outs=b_outs.to(device)

    #So first thing we need to do is to sort the enc_acts_BH by the absolute value of the features
    
    sorted_enc_vals,sorted_enc_inds=torch.sort(torch.abs(enc_acts_BH),dim=-1,descending=True)
    
    #b_dec_PBHD=b_dec_PD[:,sorted_enc_inds,:]
    
    
    # Find the largest index in dim=1 that is not zero for each element in dim=0
    #Ah that's clever - because non zero elements are ones you can just sum
    #to ge the largest value!
    
    #Note - hopefully, this filtering step is cheap, so you can always do it first
    #and then you can do filtering on the pushed through activations, too
    non_zero_indices = (sorted_enc_vals != 0).sum(dim=1)
    # Get the maximum index across all elements in dim=0
    max_non_zero_index = non_zero_indices.max().item()

    filtered_sorted_enc_BH=sorted_enc_vals[:,:max_non_zero_index]
    #print(f'filtered_sorted_enc_BH.shape: {sorted_enc_inds[:,:max_non_zero_index].shape}')
    #print(f'W_dec_PHD.shape: {W_dec_PHD.shape}')
    #That doesn't make any sense?
    W_dec_PBHD=W_dec_PHD[:,sorted_enc_inds[:,:max_non_zero_index],:]
    #filtered_W_dec_PHD=W_dec_PHD[:,:max_non_zero_index,:]
    
    
    enc_BHD_W = einops.einsum(filtered_sorted_enc_BH[...,None], W_dec_PBHD, "batch hidden_c one, block batch hidden_c d_model -> block batch d_model hidden_c")
    #print(f'enc_BHD_W.shape: {enc_BHD_W.shape}')
    enc_BHD_b = bias*b_dec_PD[:,None,:,None]/hidden_dim
    #print(f'enc_BHD_b.shape: {enc_BHD_b.shape}')
    p_BNH = einops.einsum(W_ins, enc_BHD_W+enc_BHD_b, "block d_model d_mlp, block batch d_model hidden -> block batch d_mlp hidden")
    #print(f'p_BNH.shape: {p_BNH.shape}')
    #print(f'b_ins.shape: {b_ins.shape}')
    p_BNH += bias*b_ins[:,None,:,None]/hidden_dim
    #print(f'p_BNH.shape: {p_BNH.shape}')
    #OK, now I want to reindex the cutoff indices to the original indices
    
    
    return p_BNH,sorted_enc_inds
    

def add_penalty(enc_acts_BH: torch.Tensor, W_dec_PHD: torch.Tensor, b_dec_PD: torch.Tensor, 
                W_ins: torch.Tensor, b_ins: torch.Tensor, W_outs: torch.Tensor, b_outs: torch.Tensor, 
                device: str, bias: float, penalty_fn) -> torch.Tensor:
    """
    Compute penalty by processing blocks one at a time to reduce memory usage.
    
    Args:
        enc_acts_BH: Encoder activations (batch, hidden)
        W_dec_PHD, b_dec_PD: Decoder weights and biases for all blocks
        W_ins, b_ins, W_outs, b_outs: MLP weights and biases for all blocks
        device: Device to compute on
        bias: Bias term
        penalty_fn: Function to compute penalty from preactivation tensor
        
    Returns:
        Mean penalty across all blocks
    """
    n_blocks = W_dec_PHD.shape[0]
    
    # MAJOR OPTIMIZATION: Compute sorting once and reuse across all blocks (12x speedup)
    sorted_enc_vals, sorted_enc_inds = torch.sort(torch.abs(enc_acts_BH), dim=-1, descending=True)
    non_zero_indices = (sorted_enc_vals != 0).sum(dim=1)
    max_non_zero_index = non_zero_indices.max().item()
    precomputed_sort = (sorted_enc_vals, sorted_enc_inds, max_non_zero_index)
    
    # OPTIMIZATION 5: Accumulate penalty sum instead of storing all penalties
    penalty_sum = 0.0
    
    for block_idx in range(n_blocks):
        # Process one block at a time to save memory, reusing precomputed sort
        p_BNH_single, _ = get_neuron_preacts_cutoff(
            enc_acts_BH, W_dec_PHD, b_dec_PD, W_ins, b_ins, W_outs, b_outs,
            device=device, bias=bias, block_idx=block_idx, precomputed_sort=precomputed_sort
        )
        
        # Compute penalty for this block and accumulate
        block_penalty = penalty_fn(p_BNH_single)
        penalty_sum += block_penalty.item() if hasattr(block_penalty, 'item') else block_penalty
        
        # OPTIMIZATION 6: Only clear cache every few blocks to reduce overhead
        if (block_idx + 1) % 4 == 0 and device != "cpu":
            torch.cuda.empty_cache()
    
    # Return mean penalty across all blocks as tensor
    return torch.tensor(penalty_sum / n_blocks, device=enc_acts_BH.device)


def download_wandb_artifact(artifact_path, save_dir=None):
    """
    Download an artifact from Weights & Biases.
    
    Args:
        artifact_path (str): Path to the artifact in format 'entity/project/artifact:alias'
        save_dir (str, optional): Directory to save the artifact. Defaults to a timestamped folder.
    
    Returns:
        str: Path to the downloaded artifact
    """
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"wandb_artifacts_{timestamp}"
    
    # Initialize wandb API
    api = wandb.Api()
    
    try:
        # Get the artifact
        artifact = api.artifact(artifact_path)
        print(f"Found artifact: {artifact.name}")

        artifact_dir = artifact.download(root=save_dir)
        print(f"Downloaded artifact to: {artifact_dir}")
        return Path(artifact_dir)

    except Exception as e:
        print(f"Error downloading artifact: {e}")
        return None


def load_crosscoder_from_wandb(
        entity: str,
        project: str,
        run_name: str,
        save_dir: str,
        device: torch.device) -> AcausalCrosscoder:
    """
    Load a crosscoder from a wandb run.
    
    Args:
        run_path (str): Path to the run in format 'entity/project/run_id'
        device (torch.device): Device to load the model on
    
    Returns:
        AcausalCrosscoder: The loaded crosscoder
    """
    artifact_path = download_wandb_artifact(
        f"{entity}/{project}/model-checkpoint_run-{run_name}:latest",
        save_dir
    )
    return AcausalCrosscoder.load(artifact_path / "model").to(device)