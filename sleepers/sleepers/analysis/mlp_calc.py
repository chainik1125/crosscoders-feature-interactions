import numpy as np
from matplotlib import pyplot as plt
import sys
print(sys.executable)
from typing import List, Any
from pathlib import Path
import torch
import torch.nn as nn
from einops import rearrange, einsum
import einops
import wandb
from pympler import asizeof
from transformers.activations import NewGELUActivation, FastGELUActivation


from sleepers.scripts.utils import load_crosscoder_from_wandb,calculate_fvu_X
from sleepers.scripts.llms import build_llm_lora
from datasets import load_dataset
from model_diffing.utils import calculate_reconstruction_loss,l2_norm
from sleepers.scripts.utils import sharpness_func


import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sleepers.scripts.utils import sharpness_func,get_neuron_preacts
from tqdm import tqdm
import time
def get_heatmap_NH(tensor_NH:torch.Tensor, abs=True):
	
	#values_NH,indices_NH=torch.sort(tensor_NH.abs().sum(dim=0),dim=-1,descending=True)
	values_NH, indices_NH = torch.sort(tensor_NH, dim=-1, descending=True)
	if abs:
		cumsum_values_NH = torch.cumsum(values_NH, dim=-1)
	else:
		cumsum_values_NH = torch.cumsum(values_NH, dim=-1)
	
	cumsum_abs_ratio = cumsum_values_NH/cumsum_values_NH[:,-1].unsqueeze(-1)
	cumsum_signed_ratio = torch.cumsum(values_NH, dim=-1)/torch.sum(values_NH, dim=-1).unsqueeze(-1)
	print(f'cumsum abs ratio shape {cumsum_abs_ratio.shape}')
	
	#To sort the neurons by which is the most peaked
	n_vals, n_idx = torch.sort(cumsum_abs_ratio[:,0], dim=0, descending=True)
	cumsum_abs_ratio = cumsum_abs_ratio[n_idx,:]
	
	return cumsum_abs_ratio

def add_heatmap(fig:go.Figure,data_tensor_NH:torch.Tensor,row:int,col:int):
	fixed_colorscale=[[0, 'blue'],[0.5,'white'],[0.51,'green'],[0.52,'white'], [0.9, 'red'], [1, 'black']]
	if row==1 and col==1:
		show_scale=True
	else:
		show_scale=False
	fig.add_trace(
		go.Heatmap(z=data_tensor_NH,x=np.arange(data_tensor_NH.shape[1]),y=np.arange(data_tensor_NH.shape[0]),
			colorscale=fixed_colorscale, # Adjust scale to make red more prominent
			colorbar=dict(title='Color Scale'),
			showscale=show_scale,
			zmin=0,
			zmax=1
		),row=row, col=col
	)
	fig.update_xaxes(title_text="Feature")
	fig.update_yaxes(title_text="Neuron")


def data_decomposition_heatmap(enc_acts:torch.Tensor,llm:object,W_dec_HXD:torch.Tensor,b_dec_XD:torch.Tensor,block:int=0):
	
	W_in = llm.blocks[block].mlp.W_in
	b_in = llm.blocks[block].mlp.b_in
	#preacts_BNH = get_preacts_nocontract(enc_acts,W_dec_HMLD, b_dec_MLD, W_in, b_in,bias=True,block=block)
	p_BNH, data_ind_w, data_ind_b = get_preacts_ind(enc_acts, W_in, b_in, W_dec_HXD, b_dec_XD, block)
	enc_acts_tensor = enc_acts.abs()

	preacts_NH = p_BNH.abs().sum(dim=0)
	enc_acts_hm = get_heatmap_NH(enc_acts_tensor)
	heatmap_whole = get_heatmap_NH(preacts_NH)
	heatmap_data_ind = get_heatmap_NH(data_ind_w.abs())

	sep_fig=make_subplots(rows=1,cols=3,subplot_titles=['Encoding distribution alpha','Data-independent distribution (W^in W^dec)','Feature distribution'])

	add_heatmap(sep_fig,enc_acts_hm,row=1,col=1)
	add_heatmap(sep_fig,heatmap_data_ind,row=1,col=2)
	add_heatmap(sep_fig,heatmap_whole,row=1,col=3)

	sep_fig.update_xaxes(range=[0,50],row=1,col=1)
	sep_fig.update_xaxes(range=[0,100],row=1,col=3)
	sep_fig.update_yaxes(title_text="Token",col=1)
	return sep_fig

def block_heatmaps(enc_acts:torch.Tensor,llm:object):
	fig=make_subplots(rows=2,cols=4)
	for block in range(4):
		W_in = llm.blocks[block].mlp.W_in
		b_in = llm.blocks[block].mlp.b_in
		preacts_BNH = get_preacts_nocontract(enc_acts,W_dec_HMLD, b_dec_MLD, W_in, b_in,bias=True,block=block)
		preacts_NH_sum = preacts_BNH.abs().sum(dim=0)
		preacts_NH_std = preacts_BNH.std(dim=0)

		add_heatmap(fig,get_heatmap_NH(preacts_NH_sum),row=1,col=1+block)
		add_heatmap(fig,get_heatmap_NH(preacts_NH_std),row=2,col=1+block)
	return fig



def mlp_reconstruction(cache:dict,block:int,llm:object):
	print(f'Activation function is: {llm.blocks[0].mlp.act_fn}')
	act_fn=llm.blocks[0].mlp.act_fn
	raw_acts_ln=cache[f"blocks.{block}.ln2.hook_normalized"]
	raw_acts_ln=raw_acts_ln[0,:,:]

	raw_acts_mid=cache["blocks.0.hook_resid_mid"]
	raw_acts_mid=raw_acts_mid[0,:,:]

	raw_acts_post=cache["blocks.0.hook_resid_post"]
	raw_acts_post=raw_acts_post[0,:,:]


	W_in=llm.blocks[block].mlp.W_in
	b_in=llm.blocks[block].mlp.b_in
	W_out=llm.blocks[block].mlp.W_out
	b_out=llm.blocks[block].mlp.b_out


	preacts = raw_acts_ln @ W_in + b_in

	postacts = act_fn(preacts)
	postacts_relu=nn.ReLU()(preacts)
	postacts_gelu=NewGELUActivation()(preacts)

	ffn_output = postacts @ W_out + b_out
	ffn_output_relu=postacts_relu @ W_out + b_out
	ffn_output_gelu=postacts_gelu @ W_out + b_out

	reconstructed_post=raw_acts_mid+ffn_output
	reconstructed_post_relu=raw_acts_mid+ffn_output_relu
	reconstructed_post_gelu=raw_acts_mid+ffn_output_gelu

	diff=reconstructed_post-raw_acts_post
	diff_relu=reconstructed_post_relu-raw_acts_post
	diff_gelu=reconstructed_post_gelu-raw_acts_post
	print(f'diff sum: {100*diff.abs().sum()/raw_acts_post.abs().sum()}%')
	print(f'diff_relu sum: {100*diff_relu.abs().sum()/raw_acts_post.abs().sum()}%')
	print(f'diff_gelu sum: {100*diff_gelu.abs().sum()/raw_acts_post.abs().sum()}%')


def get_preacts_nocontract(enc_acts:torch.Tensor, W_dec_HMLD:torch.Tensor, b_dec_MLD:torch.Tensor,llm:object,block:int=0, bias=True, pythia_format=None):
	"""
	Get the preacts without contracting the feature dimension
	
	Args:
		pythia_format: If True, use Pythia hookpoint indexing (MLP input at index 2).
		               If False, use GPT-2 hookpoint indexing (MLP input at index 3).
		               Auto-detects based on model name if None.
	"""
	# Auto-detect architecture if not specified
	if pythia_format is None:
		model_name = getattr(llm.cfg, 'model_name', '')
		pythia_format = 'pythia' in model_name.lower()
	
	W_in = llm.blocks[block].mlp.W_in
	b_in = llm.blocks[block].mlp.b_in
	
	#Note that you have to divide by the number of features to get the correct bias
	if bias:
		bias_factor = 1
	else:
		bias_factor = 0
	hidden_dim = enc_acts.shape[1]
	
	# Choose hookpoint index based on architecture
	mlp_hookpoint_idx = 4*block + (2 if pythia_format else 3)
	dec_nocontract = (enc_acts[:,:,None] * W_dec_HMLD[None,:,0,mlp_hookpoint_idx,:]) + bias_factor*b_dec_MLD[0,mlp_hookpoint_idx,:].unsqueeze(0)/hidden_dim
		

	#OK so decoding is correct, then let's push through the mlp
	pre_acts=einsum(W_in, dec_nocontract, "d_model d_mlp, batch hidden d_model -> batch hidden d_mlp")
	pre_acts += bias_factor*b_in/hidden_dim
	pre_acts = rearrange(pre_acts, 'batch hidden d_mlp -> batch d_mlp hidden')
	
	return pre_acts

def get_weight_features(W_in:torch.Tensor, b_in:torch.Tensor, W_dec_HMLD:torch.Tensor, b_dec_MLD:torch.Tensor, block:int=0):
	"""
	Get W(W_f+B_D), without contracting the feature dimension
	"""

	hidden_dim = W_dec_HMLD.shape[0]
	
	W_dec_HD = W_dec_HMLD[:,0,3*block+1,:]
	b_dec_D = b_dec_MLD[0,3*block+1,:]
	
	b_dec_DH = torch.zeros_like(W_dec_HD)
	b_dec_DH[:,np.arange(W_dec_HD.shape[1])] = b_dec_D/hidden_dim
	b2 = b_dec_MLD[0,3*block+1,:].unsqueeze(0)/hidden_dim
	
	
	
	non_contract_tensor = einsum(W_in, (W_dec_HMLD[:,0,3*block+1,:] + b_dec_MLD[0,3*block+1,:].unsqueeze(0)/hidden_dim), "d_model d_mlp, hidden d_model -> d_mlp hidden")
	non_contract_tensor += b_in[:,None]/hidden_dim

	non_contract_tensor_W = einsum(W_in, W_dec_HD, "d_model d_mlp, hidden d_model -> d_mlp hidden")
	non_contract_tensor_W += b_in[:,None]/hidden_dim

	non_contract_tensor_b = einsum(W_in, b_dec_DH, "d_model d_mlp, hidden d_model -> d_mlp hidden")
	#Don't need this because you only add in the bias once
	#non_contract_tensor_b += b_in[:,None]/non_contract_tensor_b.shape[-1]

	#print(f'total same: {100*torch.sum(non_contract_tensor-non_contract_tensor_W-non_contract_tensor_b)/torch.sum(non_contract_tensor)}%')

	
	return non_contract_tensor, non_contract_tensor_W, non_contract_tensor_b

def get_preacts_ind(enc_acts:torch.Tensor, W_in:torch.Tensor, b_in:torch.Tensor, W_dec_HMLD:torch.Tensor, b_dec_MLD:torch.Tensor, block:int, pythia_format=None):
	"""
	Args:
		pythia_format: If True, use Pythia hookpoint indexing (MLP input at index 2).
		               If False, use GPT-2 hookpoint indexing (MLP input at index 3).
		               Auto-detects based on model name if None.
	"""
	# Auto-detect architecture if not specified (would need llm object for this)
	# For now, default to False to maintain backward compatibility
	if pythia_format is None:
		pythia_format = False
		
	hidden_dim = W_dec_HMLD.shape[0]
	mlp_hookpoint_idx = 4*block + (2 if pythia_format else 3)
	W_dec_HD = W_dec_HMLD[:,0,mlp_hookpoint_idx,:]
	b_dec_D = b_dec_MLD[0,mlp_hookpoint_idx,:]

	print(f'enc_acts.shape: {enc_acts[...,None].shape}')
	print(f'W_dec_HD.shape: {W_dec_HD[None,:,:].shape}')
	

	enc_BHD_W = einsum(enc_acts[...,None], W_dec_HD[None,:,:], "batch hidden one, one hidden d_model -> batch d_model hidden")
	enc_BHD_b = b_dec_D[None,:,None]/hidden_dim
	p_BNH = einsum(W_in, enc_BHD_W+enc_BHD_b, "d_model d_mlp, batch d_model hidden -> batch d_mlp hidden")
	p_BNH += b_in[:,None]/hidden_dim

	data_ind_w = einsum(W_in, W_dec_HD, "d_model d_mlp, hidden d_model -> d_mlp hidden")

	b_dec_DH = torch.zeros_like(W_dec_HD)
	b_dec_DH[:,np.arange(W_dec_HD.shape[1])] = b_dec_D/hidden_dim
	data_ind_b = einsum(W_in, b_dec_DH, "d_model d_mlp, hidden d_model -> d_mlp hidden")
	data_ind_b += b_in[:,None]/hidden_dim

	return enc_acts[:,None,:]*data_ind_w+data_ind_b, data_ind_w, data_ind_b


def check_data_decomposition(block:int,llm:object,enc_acts:torch.Tensor,W_dec_HMLD:torch.Tensor, b_dec_MLD:torch.Tensor):
	#block = 3
	W_in = llm.blocks[block].mlp.W_in
	W_out = llm.blocks[block].mlp.W_out
	b_in = llm.blocks[block].mlp.b_in
	b_out = llm.blocks[block].mlp.b_out


	preacts_BNH = get_preacts_nocontract(enc_acts,W_dec_HMLD, b_dec_MLD, W_in, b_in,bias=True,block=block)
	p_BNH, data_ind_w, data_ind_b = get_preacts_ind(enc_acts, W_in, b_in, W_dec_HMLD, b_dec_MLD, block)

	print(f'preacts_BNH equal to p_BNH: {100*torch.sum(preacts_BNH-p_BNH)/torch.sum(p_BNH)}% error')

def get_activations(input: str, model:Any, crosscoder:Any):
	crosscoder=crosscoder.to(DEVICE)
	tokens = torch.tensor(tokenizer.encode(input)[0:128])
	_, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=[
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
	"blocks.3.hook_resid_post"
	])
	activations_BSLD = torch.stack([cache[name] for name in cache.keys()], dim=2).to(DEVICE)
	print(f'activations_BSLD.shape: {activations_BSLD.shape}')
#    activations_BSLD = einsum(
#        activations_BSLD,
#        torch.tensor(cfg.norm_scaling_factors[0], device=DEVICE),
#        "b s l d, l -> b s l d")
	#activations_BSLD -= dataloader_mean_SMPD[0:tokens.shape[0],0,:,:].unsqueeze(0)
	activations_BSMLD = torch.unsqueeze(activations_BSLD, dim=2)
	
	activations_SMLD = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")
	feature_activations_SH = crosscoder._encode_BH(activations_SMLD)
	return feature_activations_SH.to(DEVICE),activations_SMLD.to(DEVICE)




def mlp_reconstruction_features(enc_acts:torch.Tensor,raw_acts:torch.Tensor,W_dec_HXD:torch.Tensor,b_dec_MXD:torch.Tensor,llm:object,block:int=0):
	W_in = llm.blocks[block].mlp.W_in
	b_in = llm.blocks[block].mlp.b_in
	W_out = llm.blocks[block].mlp.W_out
	b_out = llm.blocks[block].mlp.b_out

	post_mlp_acts=dec_acts[:,0,4*block+4,:]
	resid_acts=dec_acts[:,0,4*block+2,:]
	

	p_BNH, data_ind_w, data_ind_b = get_preacts_ind(enc_acts, W_in, b_in, W_dec_HXD, b_dec_MXD, block)
	p_BNH_summed=p_BNH.sum(dim=-1)
	p_BNH_summed=NewGELUActivation()(p_BNH_summed)
	post_mlp=p_BNH_summed@W_out+b_out
	rec_post=resid_acts+post_mlp

	rec_loss=calculate_reconstruction_loss(rec_post,post_mlp_acts)
	rel_rec_loss=rec_loss/(calculate_reconstruction_loss(post_mlp_acts,0))

	return rel_rec_loss

#Let's get decoder rel. reconstruction loss for each hookpoint


def hookpoints_losses(raw_acts:torch.Tensor,dec_acts:torch.Tensor,hookpoints:List[str]):
	#Want to calculate the rel. reconstruction, abs reconstruction, and unexplained variance for each hookpoint
	rec_losses=[]
	rel_rec_losses=[]
	unexplained_variances=[]

	for hp_ind, hookpoint in enumerate(hookpoints):
		rec_loss=calculate_reconstruction_loss(dec_acts[:,0,hp_ind,:],raw_acts[:,0,hp_ind,:])
		rel_rec_loss=rec_loss/(calculate_reconstruction_loss(raw_acts[:,0,hp_ind,:],0))
		unexplained_variance=calculate_fvu_X(raw_acts[:,0,hp_ind,:],dec_acts[:,0,hp_ind,:])
		rec_losses.append(rec_loss)
		rel_rec_losses.append(rel_rec_loss)
		unexplained_variances.append(unexplained_variance)

	return rec_losses,rel_rec_losses,unexplained_variances


#test_losses=hookpoints_losses(raw_acts,dec_acts,used_names)

def plot_losses(rec_losses:List[float],rel_rec_losses:List[float],unexplained_variances:List[float],used_names:List[str]):
	fig=make_subplots(rows=1,cols=3,subplot_titles=['Reconstruction Loss','Relative Reconstruction Loss','Unexplained Variance'])

	for hp_ind, hookpoint in enumerate(used_names):
		fig.add_trace(go.Scatter(x=np.arange(len(rec_losses)),y=rec_losses,name=f'{hookpoint} rec loss',showlegend=False),row=1,col=1)
		fig.add_trace(go.Scatter(x=np.arange(len(rel_rec_losses)),y=rel_rec_losses,name=f'{hookpoint} rel rec loss',showlegend=False),row=1,col=2)
		fig.add_trace(go.Scatter(x=np.arange(len(unexplained_variances)),y=unexplained_variances,name=f'{hookpoint} unexplained variance',showlegend=False),row=1,col=3)

		fig.update_xaxes(title_text="Hookpoint")
		fig.update_yaxes(title_text="Reconstruction Loss",col=1)
		fig.update_yaxes(title_text="Relative Reconstruction Loss",col=2)
		fig.update_yaxes(title_text="Unexplained Variance",col=3)

		return fig



def patched_model_loss(model, prompt, crosscoder, hook_names:List[str]):
	"""
		Finds the model loss when the model activations are replaced with the
		crosscoder reconstructions at a given layer index.
	"""
	tokens = model.to_tokens(prompt)[:,0:128]

	loss, cache = model.run_with_cache(tokens, names_filter=hook_names, return_type="loss")
	activations_BSLD = torch.stack([cache[name] for name in hook_names], dim=2)

	# add model dim 
	activations_BSXD = torch.unsqueeze(activations_BSLD, dim=2)
	# remove sequence dim (I'm considering each token in the sequence as a batch)
	activations_SXD = einops.rearrange(activations_BSXD, "b s m l d -> (b s) m l d")
	train_res = crosscoder.forward_train(activations_SXD)
	reconstructed_acts_BXD = train_res.output_BXD

	# reorder again to remove model dim and add sequence dim
	reconstructed_acts_BSLD = einops.rearrange(reconstructed_acts_BXD, "(b s) m l d -> b s m l d", b=1)
	reconstructed_acts_BSLD = reconstructed_acts_BSLD.squeeze(2)
   

	
	enc_acts,raw_acts=get_activations(prompt,model,crosscoder)
	print(f'enc_acts.shape: {enc_acts.shape}')
	print(f'raw_acts.shape: {raw_acts.shape}')
	dec_acts=crosscoder._forward(raw_acts).output_BXD
	print(f'dec_acts.shape: {dec_acts.shape}')
	
	rel_loss_func=calculate_reconstruction_loss(raw_acts[:,:,-1,:],dec_acts[:,:,-1,:])/calculate_reconstruction_loss(raw_acts[:,:,-1,:],0)
	print(f'rel_loss_func: {100*rel_loss_func}%')
	relative_loss=calculate_reconstruction_loss(activations_BSLD[:,:,-1,:],reconstructed_acts_BSLD[:,:,-1,:])/calculate_reconstruction_loss(activations_BSLD[:,:,-1,:],0)
	print(f'relative_loss: {100*relative_loss}%')

	# patch final layer activations into model
	def patch_fn(acts, hook):
		# extract final layer activations
		return dec_acts[:, :, -1, :]
	
	def patch_actual(acts,hook):
		rearrange_raw=rearrange(raw_acts,"s b l d -> b s l d")
		return rearrange_raw[:,:,-1,:]
	
	def patch_actual_original(acts,hook):
		return activations_BSLD[:,:,-1,:]
	
	def patch_original(acts,hook):
		return reconstructed_acts_BSLD[:,:,-1,:]
	
	print(f'raw acts shape: {patch_actual(None,None).shape}')
	print(f'activations_BSLD shape: {activations_BSLD.shape}')
	
	
	
	patched_loss = model.run_with_hooks(
		tokens,
		return_type="loss",
		fwd_hooks=[("blocks.3.hook_resid_post", patch_actual)]
	)
	return loss.item(), patched_loss.item()

if __name__ == '__main__':
	print(torch.__version__)
	torch.set_grad_enabled(False)
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(DEVICE)
	checkpoint_dir = Path('../../.checkpoints/')

	wandb_run_name = 'h4tax6ro'

	crosscoder = load_crosscoder_from_wandb(
	"dmitry2-uiuc",
	"sleeper-model-diffing",
	wandb_run_name,
	"../../.wandb_artifacts",
	DEVICE)

	W_dec_HXD = crosscoder.W_dec_HXD
	b_dec_XD = crosscoder.b_dec_XD

	print(f'decoding_weights.shape: {W_dec_HXD.shape}')
	print(f'decoding_bias.shape: {b_dec_XD.shape}')

	#load dataloader means, if analyzing those xc
	# api = wandb.Api()
	# artifact = api.artifact(f"dmitry2-uiuc/sleeper-model-diffing/dataloader-means_run-{wandb_run_name}:latest")
	# artifact_dir = Path(artifact.download(root="../../.wandb_artifacts"))
	# dataloader_mean_SMPD = torch.load(artifact_dir / "dataloader_means.pt", map_location=DEVICE)

	llm = build_llm_lora(
	base_model_repo="roneneldan/TinyStories-Instruct-33M",
	lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
	cache_dir=None,
	device=DEVICE,
	dtype=None
	)
	tokenizer = llm.tokenizer

	dataset = load_dataset('mars-jason-25/tiny_stories_instruct_sleeper_data', split='train')
	#dataset = load_dataset('mars-jason-25/processed_dolphin_IHY_sleeper_distilled_dataset', split='train')

	dataset = dataset.filter(lambda x: x['is_training'] == True)
	dataset = dataset.select(range(100))
	print(f'len(dataset): {len(dataset)}')

	# Import the asizeof package to get the actual memory size of the dataset

	# Get the size of the dataset in bytes
	dataset_size_bytes = asizeof.asizeof(dataset)

	# Convert to more readable formats
	dataset_size_mb = dataset_size_bytes / 1024 / 1024

	crosscoder_size_bytes = asizeof.asizeof(crosscoder)
	crosscoder_size_mb = crosscoder_size_bytes / 1024 / 1024

	llm_size_bytes = asizeof.asizeof(llm)
	llm_size_mb = llm_size_bytes / 1024 / 1024


	print(f'Dataset size: {dataset_size_mb:.2f} MB')
	print(f'Crosscoder size: {crosscoder_size_mb:.2f} MB')
	print(f'LLM size: {llm_size_mb:.2f} MB')


	block=0
	input = dataset[4]["text"]
	xc_train_names=[hook for block_ind in range(4) for hook in [
		f"blocks.{block_ind}.hook_resid_pre", 
		f"blocks.{block_ind}.ln1.hook_normalized", 
		f"blocks.{block_ind}.hook_resid_mid", 
		f"blocks.{block_ind}.ln2.hook_normalized"
	]]
	xc_train_names.append(f"blocks.{block}.hook_resid_post")


	print(xc_train_names)

	print(len(xc_train_names))

	xc_train_names_old=[hook for block_ind in range(4) for hook in [
		f"blocks.{block_ind}.hook_resid_pre", 
		f"blocks.{block_ind}.hook_resid_mid", 
		f"blocks.{block_ind}.hook_resid_post"
	]]

	print(f"input: \n {input}")


	used_names=xc_train_names

	enc_acts,raw_acts=get_activations(input,llm,crosscoder)
	dec_acts=crosscoder._forward(raw_acts).output_BXD


	print(f'dec_acts.shape: {dec_acts.shape}')
	print(f'raw_acts.shape: {raw_acts.shape}')
	print(f'enc_acts.shape: {enc_acts.shape}')
	
	# def get_neuron_preacts(enc_acts_BH:torch.Tensor,llm:object,crosscoder:object):
	# 	W_ins=torch.stack([llm.blocks[val].mlp.W_in for val in range(1)],dim=0)
	# 	b_ins=torch.stack([llm.blocks[val].mlp.b_in for val in range(1)],dim=0)
	# 	W_outs=torch.stack([llm.blocks[val].mlp.W_out for val in range(1)],dim=0)
	# 	b_outs=torch.stack([llm.blocks[val].mlp.b_out for val in range(1)],dim=0)
		
	# 	print(f'W_ins.shape: {W_ins.shape}')
	# 	print(f'b_ins.shape: {b_ins.shape}')
	# 	print(f'W_outs.shape: {W_outs.shape}')
	# 	print(f'b_outs.shape: {b_outs.shape}')

	# 	W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in range(1)],dim=0)
	# 	b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,4*block+3,:] for block in range(1)],dim=0)
		
	# 	hidden_dim=W_dec_PHD.shape[1]
	
	# 	enc_BHD_W = einsum(enc_acts_BH[...,None], W_dec_PHD[None,:,:], "batch hidden one, one block hidden d_model -> block batch d_model hidden")
	# 	print(f'enc_BHD_W.shape: {enc_BHD_W.shape}')
	# 	enc_BHD_b = b_dec_PD[:,None,:,None]/hidden_dim
	# 	print(f'enc_BHD_b.shape: {enc_BHD_b.shape}')
	# 	p_BNH = einsum(W_ins, enc_BHD_W+enc_BHD_b, "block d_model d_mlp, block batch d_model hidden -> block batch d_mlp hidden")
	# 	print(f'p_BNH.shape: {p_BNH.shape}')
	# 	print(f'b_ins.shape: {b_ins.shape}')
		
	# 	p_BNH += b_ins[:,None,:,None]/hidden_dim
	# 	print(f'p_BNH.shape: {p_BNH.shape}')
		
		
		
	# 	return p_BNH
	#test_sharps=neuron_sharpness_full(enc_acts,llm,crosscoder)
	def neuron_sharpness_full(enc_acts_BH:torch.Tensor,llm:object,crosscoder:object):
		W_ins=torch.stack([llm.blocks[val].mlp.W_in for val in range(4)],dim=0)
		b_ins=torch.stack([llm.blocks[val].mlp.b_in for val in range(4)],dim=0)
		W_outs=torch.stack([llm.blocks[val].mlp.W_out for val in range(4)],dim=0)
		b_outs=torch.stack([llm.blocks[val].mlp.b_out for val in range(4)],dim=0)
		
		W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in range(4)],dim=0)
		b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
		
		sharps = []
		chunk_size = 8 # You can adjust this parameter as needed

		for batch_start in tqdm(range(0, enc_acts.shape[0], chunk_size)):
			# Get the current chunk (handle the case where the last chunk might be smaller)
			batch_end = min(batch_start + chunk_size, enc_acts.shape[0])
			current_chunk = enc_acts[batch_start:batch_end]
			
			p_PBNH=get_neuron_preacts(current_chunk,W_dec_PHD,b_dec_PD,W_ins,b_ins,W_outs,b_outs,DEVICE)
			first=True
			if first:
				first=False
				print(f'p_PHND.shape: {p_PBNH.shape}')
			
			sharp=sharpness_func(p_PBNH)
			sharps.append(sharp)
		return np.mean(np.array(sharps))
	
	def get_neuron_preacts_cutoff(enc_acts_BH:torch.Tensor,W_dec_PHD:torch.Tensor,b_dec_PD:torch.Tensor,W_ins:torch.Tensor,b_ins:torch.Tensor,W_outs:torch.Tensor,b_outs:torch.Tensor,device:str="cpu"):
		"""
		Idea of this calculation is to cutoff the encoding past the point
		where the contributions are negligible.
		"""

		hidden_dim=W_dec_PHD.shape[1]
		enc_acts_BH=enc_acts_BH.to(device)
		W_dec_PHD=W_dec_PHD.to(device)
		b_dec_PD=b_dec_PD.to(device)
		W_ins=W_ins.to(device)
		b_ins=b_ins.to(device)
		W_outs=W_outs.to(device)
		b_outs=b_outs.to(device)

		#So first thing we need to do is to sort the enc_acts_BH by the absolute value of the features
		start_time=time.time()
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
		W_dec_PBHD=W_dec_PHD[:,sorted_enc_inds[:,:max_non_zero_index],:]
		#filtered_W_dec_PHD=W_dec_PHD[:,:max_non_zero_index,:]

		enc_BHD_W = einops.einsum(filtered_sorted_enc_BH[...,None], W_dec_PBHD, "batch hidden_c one, block batch hidden_c d_model -> block batch d_model hidden_c")
		#print(f'enc_BHD_W.shape: {enc_BHD_W.shape}')
		enc_BHD_b = b_dec_PD[:,None,:,None]/hidden_dim
		#print(f'enc_BHD_b.shape: {enc_BHD_b.shape}')
		p_BNH = einops.einsum(W_ins, enc_BHD_W+enc_BHD_b, "block d_model d_mlp, block batch d_model hidden -> block batch d_mlp hidden")
		#print(f'p_BNH.shape: {p_BNH.shape}')
		#print(f'b_ins.shape: {b_ins.shape}')
		
		p_BNH += b_ins[:,None,:,None]/hidden_dim
		#print(f'p_BNH.shape: {p_BNH.shape}')
		end_time=time.time()
		print(f'time taken: {end_time-start_time} seconds')
		return p_BNH
	
	W_ins=torch.stack([llm.blocks[val].mlp.W_in for val in range(4)],dim=0)
	b_ins=torch.stack([llm.blocks[val].mlp.b_in for val in range(4)],dim=0)
	W_outs=torch.stack([llm.blocks[val].mlp.W_out for val in range(4)],dim=0)
	b_outs=torch.stack([llm.blocks[val].mlp.b_out for val in range(4)],dim=0)
	
	W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in range(4)],dim=0)
	b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
	
	test_cutoff=get_neuron_preacts_cutoff(enc_acts,W_dec_PHD,b_dec_PD,W_ins,b_ins,W_outs,b_outs,DEVICE)
	print(f'test_cutoff.shape: {test_cutoff.shape}')

	exit()
	
	def neuron_sharpness_full_2(enc_acts_BH:torch.Tensor,llm:object,crosscoder:object):
		W_ins=torch.stack([llm.blocks[val].mlp.W_in for val in range(4)],dim=0)
		b_ins=torch.stack([llm.blocks[val].mlp.b_in for val in range(4)],dim=0)
		W_outs=torch.stack([llm.blocks[val].mlp.W_out for val in range(4)],dim=0)
		b_outs=torch.stack([llm.blocks[val].mlp.b_out for val in range(4)],dim=0)
		
		W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in range(4)],dim=0)
		b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
		
		sharps = []
		chunk_size = 8 # You can adjust this parameter as needed

		for batch_start in tqdm(range(0, enc_acts.shape[0], chunk_size)):
			# Get the current chunk (handle the case where the last chunk might be smaller)
			batch_end = min(batch_start + chunk_size, enc_acts.shape[0])
			current_chunk = enc_acts[batch_start:batch_end]
			
			p_PBNH=get_neuron_preacts(current_chunk,W_dec_PHD,b_dec_PD,W_ins,b_ins,W_outs,b_outs,DEVICE)
			first=True
			if first:
				first=False
				print(f'p_PHND.shape: {p_PBNH.shape}')
			
			sharp=sharpness_func(p_PBNH)
			sharps.append(sharp)
		return np.mean(np.array(sharps))
	

	exit()


	def neuron_sharpness_quick(enc_acts_BH:torch.Tensor,llm:object,crosscoder:object):
		W_ins=torch.stack([llm.blocks[val].mlp.W_in for val in range(4)],dim=0).to(DEVICE)
		b_ins=torch.stack([llm.blocks[val].mlp.b_in for val in range(4)],dim=0).to(DEVICE)	
		W_outs=torch.stack([llm.blocks[val].mlp.W_out for val in range(4)],dim=0).to(DEVICE)	
		b_outs=torch.stack([llm.blocks[val].mlp.b_out for val in range(4)],dim=0).to(DEVICE)

		crosscoder=crosscoder.to(DEVICE)
		
		W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in range(4)],dim=0)#.to(DEVICE)
		b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)#.to(DEVICE)
		
		time_start=time.time()
		enc_weights=enc_acts_BH[enc_acts_BH != 0].mean(dim=0)
		data_ind_w=einops.einsum(W_ins,W_dec_PHD,"block d_model d_mlp, block hidden d_model -> block d_mlp hidden",)
		data_ind_b=b_ins[:,:,None]/W_dec_PHD.shape[1]
		data_weighted_w=enc_weights.unsqueeze(0).unsqueeze(0)*data_ind_w
	

		
		#p_PNH=data_ind_w+0*data_ind_b
		p_PNH=data_weighted_w+0*data_ind_b
		
		
		sharp=sharpness_func(p_PNH)
		time_end=time.time()
		print(f'time taken: {time_end-time_start} seconds')
		return sharp
	
	test_sharp=neuron_sharpness_quick(enc_acts,llm,crosscoder)
	print(f'test_sharp: {test_sharp}')

	exit()

	
	rec_loss=calculate_reconstruction_loss(raw_acts[:,0,:,:],dec_acts[:,0,:,:])
	print(f'rec_loss: {rec_loss}')
	rec_loss=calculate_reconstruction_loss(raw_acts[:,0,:,:],dec_acts[:,0,:,:])
	print(f'rec_loss: {rec_loss}')
	print(f'rec_loss percent: {100*rec_loss/(calculate_reconstruction_loss(raw_acts[:,0,:,:],0))}%')

	#get the unexplained variance
	unexplained_variance = []
	for p in range(raw_acts.shape[2]):
		unexplained_variance.append(calculate_fvu_X(raw_acts[:,:,p,:], dec_acts[:,:,p,:]).cpu().detach().numpy())
	mean_unexplained_variance = np.mean(np.array(unexplained_variance))

	print(f'mean unexplained variance: {mean_unexplained_variance}')

	#test_losses=hookpoints_losses(raw_acts,dec_acts,used_names)
	#plot_losses(*test_losses,used_names).show()

	#data_decomposition_heatmap(enc_acts,llm,W_dec_HXD,b_dec_XD,0).show()

	patched_losses=patched_model_loss(llm,input,crosscoder,used_names)
	print(f'patched_losses: {patched_losses}')

	




