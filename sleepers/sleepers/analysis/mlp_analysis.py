import numpy as np
from matplotlib import pyplot as plt
import sys

from datetime import datetime
import pickle
import os
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
from sleepers.scripts.utils import get_neuron_preacts_cutoff
from sleepers.analysis.ft_analysis_util import display_feature_activation_visualization
from sleepers.analysis.analysis_utils import serve_multiple_visualizations
from sleepers.analysis.analysis_utils import save_dict,load_dict, get_preacts_nocontract, get_activations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sleepers.scripts.utils import get_neuron_preacts
from tqdm import tqdm
from collections import defaultdict
import pickle
from datetime import datetime
import random



def get_heatmap_BNH_datafirst(tensor_BNH:torch.Tensor,abs=True):
	#values_NH,indices_NH=torch.sort(tensor_BNH.abs().sum(dim=0),dim=-1,descending=True)
	values_BNH, indices_BNH = torch.sort(tensor_BNH, dim=-1, descending=True)
	if abs:
		cumsum_values_BNH = torch.cumsum(values_BNH, dim=-1)
	else:
		cumsum_values_BNH = torch.cumsum(values_BNH, dim=-1)
	
	cumsum_abs_ratio = cumsum_values_BNH/cumsum_values_BNH[:,:,-1].unsqueeze(-1)
	cumsum_signed_ratio = torch.cumsum(values_BNH, dim=-1)/torch.sum(values_BNH, dim=-1).unsqueeze(-1)
	print(f'cumsum abs ratio shape {cumsum_abs_ratio.shape}')
	
	#To sort the neurons by which is the most peaked
	n_vals, n_idx = torch.sort(cumsum_abs_ratio[:,:,0], dim=1, descending=True)
	cumsum_abs_ratio = cumsum_abs_ratio[:,n_idx,:]

	cumsum_abs_ratio_datamean=cumsum_abs_ratio.mean(dim=0)
	
	return cumsum_abs_ratio

def get_heatmap_NH(tensor_NH:torch.Tensor,sort_max=True):
	
	#values_NH,indices_NH=torch.sort(tensor_NH.abs().sum(dim=0),dim=-1,descending=True)
	values_NH, indices_NH = torch.sort(tensor_NH, dim=-1, descending=True)
	
	cumsum_values_NH = torch.cumsum(values_NH, dim=-1)
	
	cumsum_abs_ratio = cumsum_values_NH/cumsum_values_NH[:,-1].unsqueeze(-1)
	#cumsum_signed_ratio = torch.cumsum(values_NH, dim=-1)/torch.sum(values_NH, dim=-1).unsqueeze(-1)
	#print(f'cumsum abs ratio shape {cumsum_abs_ratio.shape}')
	
	#To sort the neurons by which is the most peaked
	if sort_max:
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
			#colorbar=dict(title='Color Scale'),
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

def data_decomposition_heatmap_comparison(input:str,llm:object,crosscoder_new:object,crosscoder_old:object,block:int=0):
	
	W_in = llm.blocks[block].mlp.W_in
	b_in = llm.blocks[block].mlp.b_in
	W_dec_HXD_new = crosscoder_new.W_dec_HXD
	b_dec_XD_new = crosscoder_new.b_dec_XD
	W_dec_HXD_old = crosscoder_old.W_dec_HXD
	b_dec_XD_old = crosscoder_old.b_dec_XD
	
	enc_acts_new,raw_acts_new=get_activations(input,llm,crosscoder_new)
	enc_acts_old,raw_acts_old=get_activations(input,llm,crosscoder_old)
	#preacts_BNH = get_preacts_nocontract(enc_acts,W_dec_HMLD, b_dec_MLD, W_in, b_in,bias=True,block=block)

	p_BNH, data_ind_w, data_ind_b = get_preacts_ind(enc_acts_new, W_in, b_in, W_dec_HXD_new, b_dec_XD_new, block)
	enc_acts_tensor = enc_acts.abs()

	preacts_NH = p_BNH.abs().sum(dim=0)
	enc_acts_hm = get_heatmap_NH(enc_acts_tensor)
	heatmap_whole = get_heatmap_NH(preacts_NH)
	heatmap_data_ind = get_heatmap_NH(data_ind_w.abs())

	sep_fig=make_subplots(rows=2,cols=3,subplot_titles=['Feature encodings','Decoder norms at MLP','Features at MLP'],vertical_spacing=0.1)

	add_heatmap(sep_fig,enc_acts_hm,row=1,col=1)
	add_heatmap(sep_fig,heatmap_data_ind,row=1,col=2)
	add_heatmap(sep_fig,heatmap_whole,row=1,col=3)

	p_BNH_old,data_ind_w_old,data_ind_b_old = get_preacts_ind(enc_acts_old, W_in, b_in, W_dec_HXD_old, b_dec_XD_old, block)
	preacts_NH_old = p_BNH_old.abs().sum(dim=0)

	enc_acts_hm_old = get_heatmap_NH(enc_acts_old.abs())
	heatmap_data_ind_old = get_heatmap_NH(data_ind_w_old.abs())
	heatmap_whole_old = get_heatmap_NH(preacts_NH_old)

	add_heatmap(sep_fig,enc_acts_hm_old,row=2,col=1)
	add_heatmap(sep_fig,heatmap_data_ind_old,row=2,col=2)
	add_heatmap(sep_fig,heatmap_whole_old,row=2,col=3)
	
	
	

	sep_fig.update_xaxes(range=[0,50],col=1)
	sep_fig.update_xaxes(range=[0,100],col=3)
	sep_fig.update_xaxes(range=[0,250],col=2)
	sep_fig.update_yaxes(title_text="Token",col=1)
	return sep_fig


def data_decomposition_heatmap_comparison_datalast(input:str,llm:object,crosscoder_new:object,crosscoder_old:object,block:int=0,bias:bool=True):
	
	W_in = llm.blocks[block].mlp.W_in
	b_in = llm.blocks[block].mlp.b_in
	W_dec_HXD_new = crosscoder_new.W_dec_HXD
	b_dec_XD_new = crosscoder_new.b_dec_XD

	W_dec_HXD_old = crosscoder_old.W_dec_HXD
	b_dec_XD_old = crosscoder_old.b_dec_XD
	
	enc_acts_new,raw_acts_new=get_activations(input,llm,crosscoder_new)
	enc_acts_old,raw_acts_old=get_activations(input,llm,crosscoder_old)
	#preacts_BNH = get_preacts_nocontract(enc_acts,W_dec_HMLD, b_dec_MLD, W_in, b_in,bias=True,block=block)

	p_BNH, data_ind_w, data_ind_b = get_preacts_ind(enc_acts_new, W_in, b_in, W_dec_HXD_new, b_dec_XD_new, block)
	p_BNH=get_preacts_nocontract(enc_acts_new,W_dec_HXD_new,b_dec_XD_new,llm,bias=bias,block=block)

	
	enc_acts_tensor = enc_acts.abs()

	
	enc_acts_hm = get_heatmap_NH(enc_acts_tensor)
	
	heatmap_data_ind = get_heatmap_NH(data_ind_w.abs())

	#preacts_NH = p_BNH.abs().sum(dim=0)
	datalast_heatmap_whole=torch.zeros((p_BNH.shape[1],p_BNH.shape[2]))
	for data_ind in tqdm(range(p_BNH.shape[0])):#
		heatmap_whole = get_heatmap_NH(p_BNH[data_ind,:,:].abs(),sort_max=True)
		datalast_heatmap_whole+=heatmap_whole
	datalast_heatmap_whole /= p_BNH.shape[0]

	
	
		
	

	sep_fig=make_subplots(rows=2,cols=3,subplot_titles=['Feature encodings','Decoder norms at MLP','Features at MLP'],vertical_spacing=0.1)

	add_heatmap(sep_fig,enc_acts_hm,row=1,col=1)
	add_heatmap(sep_fig,heatmap_data_ind,row=1,col=2)
	add_heatmap(sep_fig,datalast_heatmap_whole,row=1,col=3)

	p_BNH_old,data_ind_w_old,data_ind_b_old = get_preacts_ind(enc_acts_old, W_in, b_in, W_dec_HXD_old, b_dec_XD_old, block)
	p_BNH_old=get_preacts_nocontract(enc_acts_new,W_dec_HXD_old,b_dec_XD_old,llm,bias=bias,block=block)
	enc_acts_hm_old = get_heatmap_NH(enc_acts_old.abs())
	heatmap_data_ind_old = get_heatmap_NH(data_ind_w_old.abs())

	datalast_heatmap_whole_old=torch.zeros((p_BNH.shape[1],p_BNH.shape[2]))
	for data_ind in tqdm(range(p_BNH_old.shape[0])):#:
		heatmap_whole = get_heatmap_NH(p_BNH_old[data_ind,:,:].abs(),sort_max=True)
		datalast_heatmap_whole_old+=heatmap_whole
	datalast_heatmap_whole_old /= p_BNH_old.shape[0]
	

	add_heatmap(sep_fig,enc_acts_hm_old,row=2,col=1)
	add_heatmap(sep_fig,heatmap_data_ind_old,row=2,col=2)
	add_heatmap(sep_fig,datalast_heatmap_whole_old,row=2,col=3)

	

	sep_fig.update_xaxes(range=[0,50],col=1)
	sep_fig.update_xaxes(range=[0,100],col=3)
	sep_fig.update_xaxes(range=[0,250],col=2)
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

def get_preacts_ind(enc_acts:torch.Tensor, W_in:torch.Tensor, b_in:torch.Tensor, W_dec_HMLD:torch.Tensor, b_dec_MLD:torch.Tensor, block:int):
		
	hidden_dim = W_dec_HMLD.shape[0]
	W_dec_HD = W_dec_HMLD[:,0,4*block+3,:]
	b_dec_D = b_dec_MLD[0,4*block+3,:]

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

# old get_activations
# def get_activations(input: str, model:Any, crosscoder:Any):
# 	tokens = torch.tensor(model.tokenizer.encode(input)[0:128])
# 	print(f'tokens.shape: {tokens.shape}')
# 	_, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=[
# 	"blocks.0.hook_resid_pre",
# 	"blocks.0.ln1.hook_normalized", 
# 	"blocks.0.hook_resid_mid",
# 	"blocks.0.ln2.hook_normalized",
# 	"blocks.1.hook_resid_pre", 
# 	"blocks.1.ln1.hook_normalized",
# 	"blocks.1.hook_resid_mid", 
# 	"blocks.1.ln2.hook_normalized",
# 	"blocks.2.hook_resid_pre",
# 	"blocks.2.ln1.hook_normalized", 
# 	"blocks.2.hook_resid_mid",
# 	"blocks.2.ln2.hook_normalized", 
# 	"blocks.3.hook_resid_pre",
# 	"blocks.3.ln1.hook_normalized",
# 	"blocks.3.hook_resid_mid", 
# 	"blocks.3.ln2.hook_normalized",
# 	"blocks.3.hook_resid_post"
# 	])
# 	activations_BSLD = torch.stack([cache[name] for name in cache.keys()], dim=2)
# 	#print(f'activations_BSLD.shape: {activations_BSLD.shape}')
# #    activations_BSLD = einsum(
# #        activations_BSLD,
# #        torch.tensor(cfg.norm_scaling_factors[0], device=DEVICE),
# #        "b s l d, l -> b s l d")
# 	#activations_BSLD -= dataloader_mean_SMPD[0:tokens.shape[0],0,:,:].unsqueeze(0)
# 	activations_BSMLD = torch.unsqueeze(activations_BSLD, dim=2)
	
# 	activations_SMLD = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")
# 	feature_activations_SH = crosscoder._encode_BH(activations_SMLD)
# 	return feature_activations_SH,activations_SMLD










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
   

	
	enc_acts,raw_acts=get_activations(prompt,model,crosscoder,tokenizer)
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

def sharpness_cdf(dataset:Any,crosscoder:Any,llm:Any):
	W_ins=torch.stack([llm.blocks[block].mlp.W_in for block in range(4)],dim=0)
	b_ins=torch.stack([llm.blocks[block].mlp.b_in for block in range(4)],dim=0)
	W_outs=torch.stack([llm.blocks[block].mlp.W_out for block in range(4)],dim=0)
	b_outs=torch.stack([llm.blocks[block].mlp.b_out for block in range(4)],dim=0)

	W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in range(4)],dim=0)
	b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)

	neuron_dim=W_ins.shape[-1]
	feature_dim=W_dec_PHD.shape[1]

	
	sharpness_tensor=torch.zeros(feature_dim)
	sharpness_maxes=torch.zeros(feature_dim)
	sharpness_mins=torch.ones(feature_dim)*float('inf')
	sharpness_squares=torch.zeros(feature_dim)
	count=0
	for story_ind,story in tqdm(enumerate(dataset)):
		input=story["text"]
		enc_acts,raw_acts=get_activations(input,llm,crosscoder)
		#print(f'enc_acts.shape: {enc_acts.shape}')
		#print(f'raw_acts.shape: {raw_acts.shape}')
		for data_ind in range(enc_acts.shape[0]):
			preacts_LBNH=get_neuron_preacts(enc_acts[data_ind,:].unsqueeze(0),W_dec_PHD,b_dec_PD,W_ins,b_ins,W_outs,b_outs,device=DEVICE,bias=False)
			sorted_vals,sorted_indices=torch.sort(preacts_LBNH.abs(),dim=-1,descending=True)
			features_cumsum=torch.cumsum(sorted_vals,dim=-1)
			features_cumsum=features_cumsum/features_cumsum[:,:,:,-1].unsqueeze(-1)
			
			features_maxes=features_cumsum.amax(dim=(0,1,2))
			features_mins=features_cumsum.amin(dim=(0,1,2))
			sharpness_maxes=torch.max(sharpness_maxes,features_maxes)
			sharpness_mins=torch.min(sharpness_mins,features_mins)
			
			features_squares=features_cumsum.pow(2).sum(dim=(0,1,2))
			features_cumsum=features_cumsum.sum(dim=(0,1,2))
			


			sharpness_tensor+=features_cumsum
			sharpness_squares+=features_squares
			count+=1
		current_average=sharpness_tensor/count
		# print(f'current_average first 10: {current_average[:10]}')
	sharpness_tensor=sharpness_tensor/(count*torch.numel(preacts_LBNH[:,:,:,0]))
	sharpness_squares=sharpness_squares/(count*torch.numel(preacts_LBNH[:,:,:,0]))
	sharpness_tensor=sharpness_tensor.cpu().detach().numpy()
	sharpness_std=(sharpness_squares-sharpness_tensor**2).pow(1/2)
	sharpness_std=sharpness_std.cpu().detach().numpy()
	return sharpness_tensor,sharpness_std,sharpness_maxes,sharpness_mins
	
		
def sharpness_histogram(dataset: Any, crosscoder: Any, llm: Any, threshold: float = 0.95, num_bins: int = 50, max_features: int = 50):
	"""
	Calculate a histogram of the number of features required to exceed a threshold of the absolute sum.
	
	Args:
		dataset: The dataset to process
		crosscoder: The crosscoder model
		llm: The language model
		threshold: The threshold percentage of the absolute sum (e.g., 0.95 for 95%)
		num_bins: Number of histogram bins
		max_features: Maximum number of features to consider for binning
		
	Returns:
		bin_edges: The bin edges for the histogram
		hist_counts: The counts for each bin
	"""
	W_ins = torch.stack([llm.blocks[block].mlp.W_in for block in range(4)], dim=0)
	b_ins = torch.stack([llm.blocks[block].mlp.b_in for block in range(4)], dim=0)
	W_outs = torch.stack([llm.blocks[block].mlp.W_out for block in range(4)], dim=0)
	b_outs = torch.stack([llm.blocks[block].mlp.b_out for block in range(4)], dim=0)

	W_dec_PHD = torch.stack([crosscoder.W_dec_HXD[:, 0, 4*block+3, :] for block in range(4)], dim=0)
	b_dec_PD = torch.stack([crosscoder.b_dec_XD[0, 4*block+3, :] for block in range(4)], dim=0)

	feature_dim=W_dec_PHD.shape[1]
	# Define histogram bins (from 1 to max_features)
	bin_edges = torch.linspace(1, max_features, num_bins + 1)
	bin_width = (max_features - 1) / num_bins
	
	# Initialize histogram counts
	hist_counts = torch.zeros(num_bins, dtype=torch.int64)
	
	# Track total number of samples processed
	total_samples = 0
	
	for story_ind, story in tqdm(enumerate(dataset)):
		input_text = story["text"]
		enc_acts, raw_acts = get_activations(input_text, llm, crosscoder)
		
		for data_ind in range(enc_acts.shape[0]):
			# Get preactivations
			preacts_LBNH = get_neuron_preacts(
				enc_acts[data_ind, :].unsqueeze(0),
				W_dec_PHD, b_dec_PD, W_ins, b_ins, W_outs, b_outs,
				device=DEVICE, bias=False
			)
			
			# Sort the absolute values in descending order
			sorted_vals, _ = torch.sort(preacts_LBNH.abs(), dim=-1, descending=True)
			
			# Calculate cumulative sum
			cumsum_vals = torch.cumsum(sorted_vals, dim=-1)
			
			# Normalize by the total sum
			total_sums = cumsum_vals[:, :, :, -1].unsqueeze(-1)
			normalized_cumsum = cumsum_vals / total_sums
			
			# Find the number of features needed to exceed the threshold for each sample
			# For each layer, batch, and neuron, find the first index where normalized_cumsum >= threshold
			features_needed = torch.sum(normalized_cumsum < threshold, dim=-1) + 1
			
			# Flatten to get all samples
			features_needed_flat = features_needed.flatten()
			
			# Update histogram counts
			for feat_count in features_needed_flat:
				# Skip if outside our maximum range
				if feat_count > max_features:
					continue
					
				# Calculate bin index and update count
				bin_idx = min(num_bins - 1, torch.floor((feat_count - 1) / bin_width).long())
				hist_counts[bin_idx] += 1
			
			total_samples += features_needed_flat.numel()
	
	# Convert counts to frequencies (optional)
	hist_frequencies = hist_counts.float() / total_samples
	
	# Convert bin_edges and hist_counts to numpy for easier plotting
	bin_edges_np = bin_edges.cpu().numpy()
	hist_counts_np = hist_counts.cpu().numpy()
	hist_frequencies_np = hist_frequencies.cpu().numpy()
	
	return bin_edges_np, hist_counts_np, hist_frequencies_np


def make_adjacency_matrix_dominant(llm:object,enc_acts:torch.Tensor,crosscoder:object,eps:float=1e-2):
	#Let's do it across all blocks
	n_features=enc_acts.shape[1]
	n_neurons=1536
	adjacency_matrix = torch.zeros((n_features, n_features))
	W_dec_HMLD=crosscoder.W_dec_HXD
	b_dec_MLD=crosscoder.b_dec_XD
	for block in range(4):
		W_in = llm.blocks[block].mlp.W_in
		b_in = llm.blocks[block].mlp.b_in
		preacts_BNH=get_preacts_nocontract(enc_acts,W_dec_HMLD, b_dec_MLD,llm,bias=False,block=block)
		abs_vals,abs_idx=torch.sort(preacts_BNH.abs(),dim=-1,descending=True)

		
		#Take 100 largest values to try and dodge convergence issues
		top_vals_BNH,top_idx_BNH=abs_vals[:,:,:100],abs_idx[:,:,:100]		
		ratio_of_max=top_vals_BNH[:,:,1:]/top_vals_BNH[:,:,0].unsqueeze(-1)

		p_BNH, data_ind_w, data_ind_b = get_preacts_ind(enc_acts, W_in, b_in, W_dec_HMLD, b_dec_MLD, block)
		
		#I think it should be for each datapoint whats largest feature in push through?
		averaged_enc_acts_H=enc_acts.abs().mean(dim=0)
		ranked_enc_acts,ranked_enc_idx=torch.sort(averaged_enc_acts_H,dim=-1,descending=True)
		#I guess you should rank the features by the average ranking?
		avg_p_vals_NH,avg_p_idx_NH=torch.sort(p_BNH.abs().mean(dim=0),dim=-1,descending=True)
		
		data_ind_w_ranked,data_ind_b_ranked=data_ind_w[:,ranked_enc_idx],data_ind_b[:,ranked_enc_idx]

		prod_NH=averaged_enc_acts_H[ranked_enc_idx]*data_ind_w_ranked+data_ind_b_ranked

		
		s_p_NH_vals,s_p_NH_idx=torch.sort(prod_NH.abs(),dim=-1,descending=True)
		
		
		
		#I guess you can just get the top 100 ones in the abs
		prod_NH_100=prod_NH[:,:100].abs()
		#Note - the ranked id doesn't sort properly because its 
		unique_max_idx=torch.unique(torch.argmax(prod_NH_100,dim=-1))
		
		
		count_tensor=torch.zeros((len(unique_max_idx),100))
		int_metric=torch.zeros((len(unique_max_idx),100))
		for neuron_idx in range(prod_NH_100.shape[0]):
			#for j in range(prod_NH_100.shape[1]):
			max_at_neuron_val,max_at_neuron_idx=torch.max(prod_NH_100[neuron_idx],dim=-1)
			max_index=torch.where(unique_max_idx==max_at_neuron_idx)[0]
			
			int_metric[max_index,:]+=((prod_NH_100[neuron_idx,:]+eps)/(max_at_neuron_val+eps).unsqueeze(0))
			count_tensor[max_index,:]+=1
			#int_metric[idx,j]=((prod_NH_100[:,j]+eps)/(prod_NH_100[:,index]+eps)).mean(dim=0)
		
		int_metric=int_metric/count_tensor
		

		
		return int_metric

def save_data_decomp_comp(fig):
	# Reduce font size in the figure
	fig.update_layout(
		font=dict(size=10),  # Reduce the font size to 10pt
		title_font=dict(size=12),  # Slightly larger font for the title
		legend_font=dict(size=8),
	)  # Even smaller font for the legend
	# Update font size for axis labels
	fig.update_xaxes(title_font=dict(size=10)),  # Set x-axis title font size
	fig.update_yaxes(title_font=dict(size=10)),  # Set y-axis title font size
	# Update tick font size
	fig.update_xaxes(tickfont=dict(size=10)),  # Set x-axis tick font size
	fig.update_yaxes(tickfont=dict(size=10))  # Set y-axis tick font size
	
	fig.update_annotations(font_size=10)
	# Set spacing between axis labels and axis to 0
	fig.update_xaxes(title_standoff=0)
	fig.update_yaxes(title_standoff=0)
	fig.update_layout(width=800,height=400,margin=dict(l=20,r=20,t=20,b=20))
	
	# Save the figure to a PDF file
	fig_pdf_path = "/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/graphs/data_decomposition_heatmap_comparison_shorter.pdf"
	fig.write_image(fig_pdf_path)
	print(f"Figure saved to {fig_pdf_path}")

	def make_comp_histogram(dataset,crosscoder_1,crosscoder_2,llm):
		bin_edges_new,hist_counts_new,hist_frequencies_new=sharpness_histogram([dataset[a] for a in range(2)],crosscoder_1,llm)
		bin_edges_old,hist_counts_old,hist_frequencies_old=sharpness_histogram([dataset[a] for a in range(2)],crosscoder_2,llm)

		bin_centers_new = (bin_edges_new[:-1] + bin_edges_new[1:]) / 2
		bin_centers_old = (bin_edges_old[:-1] + bin_edges_old[1:]) / 2

		fig = make_subplots(rows=1, cols=1)
		fig.add_trace(go.Bar(x=bin_centers_new, y=hist_frequencies_new, name='With penalty',opacity=0.7,marker_color='blue'))
		fig.add_trace(go.Bar(x=bin_centers_old, y=hist_frequencies_old, name='No penalty',opacity=0.7,marker_color='red'))
		fig.update_xaxes(title_text="Feature")
		fig.update_yaxes(title_text="Frequency")
		return fig
	
	
def make_comp_cdf(dataset,crosscoder_1,crosscoder_2,llm):

	test_mean,test_std,test_maxes,test_mins=sharpness_cdf([dataset[a] for a in range(2)],crosscoder_1,llm)
	
	test_mean_2,test_std_2,test_maxes_2,test_mins_2=sharpness_cdf([dataset[a] for a in range(2)],crosscoder_2,llm)


	fig=make_subplots(rows=1,cols=1)
	
	# Convert tensors to numpy arrays if they aren't already
	test_maxes_np = test_maxes.cpu().detach().numpy() if isinstance(test_maxes, torch.Tensor) else test_maxes
	test_mins_np = test_mins.cpu().detach().numpy() if isinstance(test_mins, torch.Tensor) else test_mins

	test_maxes_np_2 = test_maxes_2.cpu().detach().numpy() if isinstance(test_maxes_2, torch.Tensor) else test_maxes_2
	test_mins_np_2 = test_mins_2.cpu().detach().numpy() if isinstance(test_mins_2, torch.Tensor) else test_mins_2
	
	x_values = np.arange(len(test_mean))
	
	# Add the main line
	fig.add_trace(go.Scatter(x=x_values,y=test_mean,name='sharpness cdf',mode='lines'))
	fig.add_trace(go.Scatter(x=x_values,y=test_mean_2,name='sharpness cdf no penalty',mode='lines'))
	fig.add_trace(go.Scatter(x=x_values[:40],y=np.arange(len(test_mean[:40]))*(1/40),name='Theoretical minimum',mode='lines',line=dict(color='black',dash='dash')))

	# Add error bars as a filled area
	# fig.add_trace(go.Scatter(
	# 	x=np.concatenate([x_values, x_values[::-1]]),
	# 	#y=np.concatenate([test_maxes_np, test_mins_np[::-1]]),
	# 	y=np.concatenate([test_mean+test_std, test_mean-test_std[::-1]]),
	# 	fill='toself',
	# 	fillcolor='rgba(0,100,80,0.2)',
	# 	line=dict(color='rgba(255,255,255,0)'),
	# 	hoverinfo='skip',
	# 	showlegend=False
	# ))
	# fig.add_trace(go.Scatter(
	# 	x=np.concatenate([x_values, x_values[::-1]]),
	# 	#y=np.concatenate([test_maxes_np_2, test_mins_np_2[::-1]]),
	# 	y=np.concatenate([test_mean_2+test_std_2, test_mean_2-test_std_2[::-1]]),
	# 	fill='toself',
	# 	fillcolor='rgba(0,100,80,0.2)',
	# 	line=dict(color='rgba(255,255,255,0)'),
	# 	hoverinfo='skip',
	# 	showlegend=False
	# ))
	
	fig.update_xaxes(title_text="Feature")
	fig.update_yaxes(title_text="Cumulative Sum of Absolute Values")
	return fig


def count_largest_features(dataset, llm, crosscoder, num_samples=1000, blocks_to_analyze=None):
	"""
	Count the frequency of dominant features across a dataset and compute their mean values.
	
	Args:
		dataset: The dataset to process
		llm: The language model
		crosscoder: The crosscoder model
		num_samples: Number of samples to process from the dataset
		blocks_to_analyze: List of blocks to analyze (default: all blocks)
		
	Returns:
		features_stats: Dictionary mapping feature indices to their statistics
	"""
	from collections import defaultdict
	import torch
	
	if blocks_to_analyze is None:
		blocks_to_analyze = range(4)  # Default to all 4 blocks
	
	# Initialize dictionary to store feature statistics
	features_stats = defaultdict(lambda: {
		'count': 0,
		'sum': 0.0,
		'stories': set(),
		'blocks': defaultdict(int),
		'plus': 0.0,
		'minus': 0.0,
		'token_neuron_pairs': defaultdict(set)  # Maps block -> set of (token_idx, neuron_idx) pairs
	})
	
	# Process dataset
	num_samples = min(num_samples, len(dataset))
	for story_idx in tqdm(range(num_samples), desc="Counting largest features"):
		print(f'story_idx: {story_idx}')
		story = dataset[story_idx]["text"]
		enc_acts, raw_acts = get_activations(story, llm, crosscoder)
		
		for block in blocks_to_analyze:
			# Get preactivations for this block
			preacts_BNH = get_preacts_nocontract(
				enc_acts, 
				crosscoder.W_dec_HXD, 
				crosscoder.b_dec_XD, 
				llm, 
				bias=True, 
				block=block
			)
			
			# Get absolute values
			preacts_abs = preacts_BNH.abs()
			
			# Find max feature indices for each token and neuron
			max_inds = torch.argmax(preacts_abs, dim=-1)
			
			# For each unique max index, get count and mean value
			for token_idx in range(max_inds.shape[0]):
				for neuron_idx in range(max_inds.shape[1]):
					max_feat_idx = max_inds[token_idx, neuron_idx].item()
					max_feat_val = preacts_abs[token_idx, neuron_idx, max_feat_idx].item()
					max_feat_val_signed = preacts_BNH[token_idx, neuron_idx, max_feat_idx].item()
					
					# Update statistics
					features_stats[max_feat_idx]['count'] += 1
					features_stats[max_feat_idx]['sum'] += max_feat_val
					features_stats[max_feat_idx]['stories'].add(story_idx)
					features_stats[max_feat_idx]['blocks'][block] += 1
					features_stats[max_feat_idx]['token_neuron_pairs'][block].add((token_idx, neuron_idx))
					
					if max_feat_val_signed > 0:
						features_stats[max_feat_idx]['plus'] += 1
					else:
						features_stats[max_feat_idx]['minus'] += -1
	
	# Calculate means and convert to regular dict for better serialization
	result = {}
	for feat_idx, stats in features_stats.items():
		result[feat_idx] = {
			'count': stats['count'],
			'mean_value': stats['sum'] / stats['count'] if stats['count'] > 0 else 0,
			'num_stories': len(stats['stories']),
			'blocks': dict(stats['blocks']),
			'plus': stats['plus'],
			'minus': stats['minus'],
			'token_neuron_pairs': {block: list(pairs) for block, pairs in stats['token_neuron_pairs'].items()}
		}
	
	# Sort features by count and print top features
	sorted_features = sorted(result.items(), key=lambda x: x[1]['count'], reverse=True)
	print("\nTop 10 most frequent dominant features:")
	for i, (feat_idx, stats) in enumerate(sorted_features[:10]):
		print(f"{i+1}. Feature {feat_idx}: {stats['count']} occurrences, mean value: {stats['mean_value']:.4f}, appears in {stats['num_stories']} stories")
		block_counts = sorted(stats['blocks'].items(), key=lambda x: x[1], reverse=True)
		print(f"   Most common in blocks: {', '.join([f'Block {b}: {c}' for b, c in block_counts[:2]])}")
	
	return result

def get_feature_stats(dataset, llm, crosscoder, num_samples=10):
    """
    Collect statistics about feature activations across a dataset.
    
    Args:
        dataset: Dataset to analyze
        llm: Language model
        crosscoder: Crosscoder model
        num_samples: Number of samples to process
        
    Returns:
        Dictionary of feature statistics
    """
    # Determine the device from the crosscoder model
    device = next(crosscoder.parameters()).device
    print(f"Using device: {device}")
    
    # Move llm to the same device if needed
    if next(llm.parameters()).device != device:
        llm = llm.to(device)
        print(f"Moved LLM to device: {device}")
    
    # Initialize stats dictionary with tensors on the same device
    stats = {
        'max_count': torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device),
        'min_count': torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device),
        'mean_count': torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device),
        'max_value': torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device),
        'min_value': torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device),
        'mean_value': torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device),
    }
    
    # Create a dataloader with a subset of the dataset
    num_samples = min(num_samples, len(dataset))
    dataloader = [dataset[i] for i in range(num_samples)]
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting feature stats")):
        input_text = batch["text"]
        
        # Get activations and ensure they're on the correct device
		# batch this
        enc_acts, raw_acts = get_activations(input_text, llm, crosscoder)
        enc_acts = enc_acts.to(device)
        
        # Initialize counters for this batch
        maxes_NH = torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device)		# maxes_NH: max counts for each feature in each layer
        mins_NH = torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device)			# mins_NH: min counts for each feature in each layer
        means_NH = torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device)		# means_NH: mean counts for each feature in each layer
        max_values_NH = torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device)	# max_values_NH: max values for each feature in each layer
        min_values_NH = torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device)	# min_values_NH: min values for each feature in each layer
        mean_values_NH = torch.zeros((llm.cfg.n_layers, llm.cfg.d_mlp), device=device)	# mean_values_NH: mean values for each feature in each layer
        
        # Process each layer
		# vectorise this
        for layer in range(llm.cfg.n_layers):
            # Get preactivations for this layer and ensure they're on the correct device
            preacts_BNH = get_preacts_nocontract(
                enc_acts, 
                crosscoder.W_dec_HXD.to(device), 
                crosscoder.b_dec_XD.to(device), 
                llm, 
                bias=True, 
                block=layer
            ).to(device)
            
            # Calculate statistics
            max_inds = torch.argmax(preacts_BNH.abs(), dim=-1)
            min_inds = torch.argmin(preacts_BNH.abs(), dim=-1)
            
            # Count occurrences of each feature being the max or min
            for token_idx in range(max_inds.shape[0]):
                for neuron_idx in range(max_inds.shape[1]):
                    max_feat_idx = max_inds[token_idx, neuron_idx].item()
                    min_feat_idx = min_inds[token_idx, neuron_idx].item()
                    
                    maxes_NH[layer, neuron_idx] += 1
                    mins_NH[layer, neuron_idx] += 1
                    means_NH[layer, neuron_idx] += 1
                    
                    max_val = preacts_BNH[token_idx, neuron_idx, max_feat_idx].abs().item()
                    min_val = preacts_BNH[token_idx, neuron_idx, min_feat_idx].abs().item()
                    mean_val = preacts_BNH[token_idx, neuron_idx].abs().mean().item()
                    
                    max_values_NH[layer, neuron_idx] += max_val
                    min_values_NH[layer, neuron_idx] += min_val
                    mean_values_NH[layer, neuron_idx] += mean_val
        
        # Add batch statistics to overall stats
        stats['max_count'] += maxes_NH
        stats['min_count'] += mins_NH
        stats['mean_count'] += means_NH
        stats['max_value'] += max_values_NH
        stats['min_value'] += min_values_NH
        stats['mean_value'] += mean_values_NH
    
    # Calculate averages
    for key in ['max_value', 'min_value', 'mean_value']:
        stats[key] /= stats['max_count'].clamp(min=1)  # Avoid division by zero
    
    return stats


def visualize_feature_stats(features_stats, top_n=None):
	"""
	Visualize the statistics of all features.
	
	Args:
		features_stats: Dictionary mapping feature indices to their statistics
		top_n: Number of top features to display (if None, show all features)
	
	Returns:
		fig: Plotly figure object
	"""
	# Sort features by count
	sorted_features = sorted(features_stats.items(), key=lambda x: x[1]['count'], reverse=True)
	if top_n is not None:
		top_features = sorted_features[:top_n]
	else:
		top_features = sorted_features
	
	# Extract data for plotting
	feature_indices = [f"F{idx}" for idx, _ in top_features]
	counts = [stats['count'] for _, stats in top_features]
	mean_values = [stats['mean_value'] for _, stats in top_features]
	plus_values = [stats['plus'] for _, stats in top_features]
	minus_values = [stats['minus'] for _, stats in top_features]
	
	# Create figure with 2 rows and 2 columns
	fig = make_subplots(
		rows=2, cols=2,
		specs=[[{"secondary_y": True}, {"secondary_y": False}],
			   [{"secondary_y": False}, {"secondary_y": False}]],
		subplot_titles=("Feature Frequency and Mean Value", "Positive vs Negative Activations",
					   "Unique Max Features per Token", "Unique Max Features per Neuron")
	)
	
	# Row 1, Col 1: Original frequency and mean value plot
	fig.add_trace(
		go.Bar(x=feature_indices, y=counts, name="Occurrence Count", marker_color='royalblue'),
		row=1, col=1, secondary_y=False
	)
	
	fig.add_trace(
		go.Scatter(x=feature_indices, y=mean_values, name="Mean Value", 
				  mode='lines+markers', marker_color='firebrick', line=dict(width=2)),
		row=1, col=1, secondary_y=True
	)

	# Row 1, Col 2: Plus/minus values
	fig.add_trace(
		go.Bar(x=feature_indices, y=plus_values, name="Plus", marker_color='green'),
		row=1, col=2
	)

	fig.add_trace(
		go.Bar(x=feature_indices, y=minus_values, name="Minus", marker_color='Red'),
		row=1, col=2
	)

	# Calculate token and neuron diversity using exact pairs
	token_feature_counts = defaultdict(set)  # token_idx -> set of features
	neuron_feature_counts = defaultdict(lambda: defaultdict(set))
	
	for feat_idx, stats in features_stats.items():
		for block, pairs in stats['token_neuron_pairs'].items():
			for token_idx, neuron_idx in pairs:
				token_feature_counts[token_idx].add(feat_idx)
				neuron_feature_counts[block][neuron_idx].add(feat_idx)
	
	# For tokens, we want unique features across all blocks
	token_diversity = [len(features) for features in token_feature_counts.values()]
	
	# For neurons, we want to look at each block separately since they're different neurons
	neuron_diversity = []
	for block in neuron_feature_counts:
		block_diversity = [len(features) for features in neuron_feature_counts[block].values()]
		neuron_diversity.extend(block_diversity)
	
	# Row 2, Col 1: Histogram of unique features per token
	fig.add_trace(
		go.Histogram(x=token_diversity, name="Token Feature Diversity",
					marker_color='purple', nbinsx=30),
		row=2, col=1
	)

	# Row 2, Col 2: Histogram of unique features per neuron
	fig.add_trace(
		go.Histogram(x=neuron_diversity, name="Neuron Feature Diversity",
					marker_color='orange', nbinsx=30),
		row=2, col=2
	)
	
	# Update layout
	fig.update_layout(
		title="Feature Statistics Analysis",
		height=800,  # Make the figure taller to accommodate all subplots
		showlegend=True,
		legend=dict(x=1.0, y=1.0),
	)
	
	# Update axes titles
	fig.update_xaxes(title_text="Feature", row=1, col=1)
	fig.update_xaxes(title_text="Feature", row=1, col=2)
	fig.update_xaxes(title_text="Number of Unique Features", row=2, col=1)
	fig.update_xaxes(title_text="Number of Unique Features", row=2, col=2)
	
	fig.update_yaxes(title_text="Occurrence Count", secondary_y=False, row=1, col=1)
	fig.update_yaxes(title_text="Mean Activation Value", secondary_y=True, row=1, col=1)
	fig.update_yaxes(title_text="Count", row=1, col=2)
	fig.update_yaxes(title_text="Number of Tokens", row=2, col=1)
	fig.update_yaxes(title_text="Number of Neurons", row=2, col=2)
	
	# Add mean lines and print statistics
	mean_token_diversity = np.mean(token_diversity)
	mean_neuron_diversity = np.mean(neuron_diversity)
	median_neuron_diversity = np.median(neuron_diversity)
	max_neuron_diversity = np.max(neuron_diversity)
	min_neuron_diversity = np.min(neuron_diversity)
	
	print(f"\nNeuron Feature Diversity Statistics:")
	print(f"Mean: {mean_neuron_diversity:.1f}")
	print(f"Median: {median_neuron_diversity:.1f}")
	print(f"Max: {max_neuron_diversity:.1f}")
	print(f"Min: {min_neuron_diversity:.1f}")
	
	fig.add_vline(x=mean_token_diversity, line_dash="dash", line_color="purple",
				  annotation_text=f"Mean: {mean_token_diversity:.1f}", row=2, col=1)
	fig.add_vline(x=mean_neuron_diversity, line_dash="dash", line_color="orange",
				  annotation_text=f"Mean: {mean_neuron_diversity:.1f}", row=2, col=2)

	# Print total number of unique features that appeared as max
	print(f"\nTotal number of unique features that appeared as max: {len(sorted_features)}")
	
	return fig

def visualize_feature_stats_from_tensor(feature_stats_dict: dict, top_n: int = 50):
    """
    Visualize statistics from the tensor-based feature statistics dictionary.
    
    Args:
        feature_stats_dict: Dictionary containing tensor statistics with keys:
            - max_count: Number of times a feature is max
            - signed_sum: Sum of feature values
            - absolute_sum: Sum of absolute feature values
            - squared_sum: Sum of squared feature values
            - max_plus_count: Count of positive max occurrences
            - max_minus_count: Count of negative max occurrences
        top_n: Number of top features to display
    
    Returns:
        fig: Plotly figure object
    """
    # Get shapes and device
    neurons, features = feature_stats_dict['max_count'].shape
    
    # Calculate total counts and means for sorting
    total_max_counts = (feature_stats_dict['max_count']).sum(dim=0)  # Sum across neurons
    mean_abs_values = (feature_stats_dict['max_abs_sum']+1e-6) / (feature_stats_dict['max_count']+1e-6)
    mean_abs_values = mean_abs_values.mean(dim=0)  # Average across neurons
    
    # Get indices of top features by max count
    top_indices = torch.argsort(total_max_counts, descending=True)[:top_n]
    
    # Create feature indices for plotting
    feature_indices = [f"F{idx}" for idx in top_indices.cpu().numpy()]
    
    # Create figure with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        subplot_titles=("Feature Frequency and Mean Value", 
                       "Positive vs Negative Activations",
                       "Max Features per Neuron", 
                       "Max Features per Datapoint")
    )
    
    # Row 1, Col 1: Frequency and mean value
    fig.add_trace(
        go.Bar(x=feature_indices, 
               y=total_max_counts[top_indices].cpu().numpy(), 
               name="Max Count",
               marker_color='royalblue'),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=feature_indices, 
                  y=mean_abs_values[top_indices].cpu().numpy(),
                  name="Mean Absolute Value",
                  mode='lines+markers',
                  marker_color='firebrick',
                  line=dict(width=2)),
        row=1, col=1, secondary_y=True
    )
    
    # Row 1, Col 2: Positive vs Negative activations
    pos_counts = feature_stats_dict['max_plus_count'].sum(dim=0)[top_indices].cpu().numpy()
    neg_counts = feature_stats_dict['max_minus_count'].sum(dim=0)[top_indices].cpu().numpy()
    
    fig.add_trace(
        go.Bar(x=feature_indices, y=pos_counts, name="Positive", marker_color='green'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=feature_indices, y=neg_counts, name="Negative", marker_color='red'),
        row=1, col=2
    )
    
    # Row 2, Col 1: Distribution of max features per neuron
    max_features_per_neuron = (feature_stats_dict['max_count'] > 0).sum(dim=1)
    fig.add_trace(
        go.Histogram(x=max_features_per_neuron.cpu().numpy(),
                    name="Features per Neuron",
                    marker_color='purple',
                    nbinsx=30),
        row=2, col=1
    )
    
    # Row 2, Col 2: Feature activation strength distribution
    features_per_datapoint = np.array(feature_stats_dict['max_features_per_datapoint'])
    mean_features_per_datapoint = np.mean(features_per_datapoint)
    
    fig.add_trace(
        go.Histogram(x=features_per_datapoint,
               name="Max Features per Datapoint",
               marker_color='orange'),
        row=2, col=2
    )
    
    # Add mean line to features per datapoint histogram
    fig.add_vline(x=mean_features_per_datapoint,
                  line_dash="dash",
                  line_color="orange",
                  annotation_text=f"Mean: {mean_features_per_datapoint:.1f}",
                  row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title="Feature Statistics Analysis from Tensor Data",
        legend=dict(x=1.0, y=1.0)
    )
    
    # Update axes titles
    fig.update_xaxes(title_text="Feature", row=1, col=1)
    fig.update_xaxes(title_text="Feature", row=1, col=2)
    fig.update_xaxes(title_text="Number of Unique Features", row=2, col=1)
    fig.update_xaxes(title_text="Max Features per Datapoint", row=2, col=2)
    
    fig.update_yaxes(title_text="Occurrence Count", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Mean Absolute Value", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Number of Neurons", row=2, col=1)
    fig.update_yaxes(title_text="Feature frequency", row=2, col=2)
    
    # Add mean line to the features per neuron histogram
    mean_features_per_neuron = max_features_per_neuron.float().mean()
    fig.add_vline(x=mean_features_per_neuron.item(),
                  line_dash="dash",
                  line_color="purple",
                  annotation_text=f"Mean: {mean_features_per_neuron:.1f}",
                  row=2, col=1)
    
    # Print some statistics
    print("\nFeature Statistics Summary:")
    print(f"Total unique features that appeared as max: {(total_max_counts > 0).sum().item()}")
    print(f"Mean features per neuron: {mean_features_per_neuron:.1f}")
    print(f"Max features per neuron: {max_features_per_neuron.max().item()}")
    print(f"Min features per neuron: {max_features_per_neuron.min().item()}")
    print(f"Mean features per datapoint: {mean_features_per_datapoint:.1f}")
    
    return fig




if __name__ == '__main__':
	print(torch.__version__)
	torch.set_grad_enabled(False)
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(DEVICE)
	checkpoint_dir = Path('../../.checkpoints/')

	#wandb_run_name = 'h4tax6ro'
	
	#wandb_run_name = 'eliy8ywm'#MLP int. penalty, 1536
	#wandb_run_name='fiwf9l79'#MLP int. penalty, 1536, lower loss
	#wandb_run_name='rx649y7j'#No MLP int. penalty, 1536
	#wandb_run_name='q5pkbghc'#MLP new int. penalty, 1536, lambda=0, 50k epochs (benchmark)
	#wandb_run_name='z7vicsnq'#MLP L1 penalty, 1536, lambda=1000, 50k epochs
	wandb_run_name='ep69x8cv'#MLP L1 penalty, 1536, lambda=500, 50k epochs
	wandb_run_name='xywuwzih'#test l=500
	wandb_run_name='ve9w4sf0'#new l=500
	wandb_run_name='1k68kpv5'#new l=1_000, bias=True
	
	



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
	#dataset = dataset.select(range(1000))
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
	# Count non-zero features for each token (dimension 0)
	print(f'non zero feats per token: {(enc_acts != 0).sum(dim=1)}')
	

	rec_loss=calculate_reconstruction_loss(raw_acts[:,0,:,:],dec_acts[:,0,:,:])
	print(f'rec_loss: {rec_loss}')
	print(f'rec_loss percent: {100*rec_loss/(calculate_reconstruction_loss(raw_acts[:,0,:,:],0))}%')

	#get the unexplained variance
	unexplained_variance = []
	for p in range(raw_acts.shape[2]):
		unexplained_variance.append(calculate_fvu_X(raw_acts[:,:,p,:], dec_acts[:,:,p,:]).cpu().detach().numpy())
	mean_unexplained_variance = np.mean(np.array(unexplained_variance))

	print(f'mean unexplained variance: {mean_unexplained_variance}')

	#Let's first look at the magnitude of the crosscoders biases, MLP biases

	print(f'b_dec_XD.shape: {b_dec_XD.shape}')
	xc_biases_mlp=torch.stack([b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
	model_biases_mlp=torch.stack([llm.blocks[block].mlp.b_in for block in range(4)],dim=0)
	print(f'xc_biases_mlp.shape: {xc_biases_mlp.shape}')
	print(f'model_biases_mlp.shape: {model_biases_mlp.shape}')

	#You want to compare the bias to the activations
	#you need the reconstructed activations and the raw pushed through activations.

	print(f'raw_acts.shape: {raw_acts.shape}')

	W_ins=torch.stack([llm.blocks[block].mlp.W_in for block in range(4)],dim=0)
	b_ins=torch.stack([llm.blocks[block].mlp.b_in for block in range(4)],dim=0)
	
	
	def push_to_mlp(raw_acts,W_ins,b_ins):
		raw_acts_ln2=torch.stack([raw_acts[:,0,4*block+3,:] for block in range(4)],dim=0)
		print(f'raw_acts_ln2.shape: {raw_acts_ln2.shape}')
		print(f'W_ins.shape: {W_ins.shape}')
		print(f'b_ins.shape: {b_ins.shape}')
		
		mlp_push_through=raw_acts_ln2@W_ins+b_ins[:,None,:]
		
		
		return mlp_push_through
	
	
	mlp_push_through=push_to_mlp(raw_acts,W_ins,b_ins)
	preacts_PBNH=get_preacts_nocontract(enc_acts,crosscoder.W_dec_HXD,b_dec_XD,llm,bias=True,block=1)
	# preacts_PBNH_all=get_preacts_nocontract_all(enc_acts,crosscoder,llm,bias=True)

	data_dict=get_feature_stats(dataset,llm,crosscoder,num_samples=2)
	#save_dict(data_dict,'feature_stats_stories_5')
	exit()
	
	
	max_NH=torch.max(preacts_PBNH.abs(),dim=-1)[0]
	print(f'mean max to sum new: {(max_NH.abs()/preacts_PBNH.abs().sum(dim=-1)).mean()}')

	
	data_folder_path='/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/data/features'
	filename='feature_stats_stories_5_datetime2025-04-16_15-03-53.pkl'
	tensor_data_dict=load_dict(data_folder_path+'/'+filename)

	#fig = visualize_feature_stats_from_tensor(feature_dict)
	#fig.show()

	print(f'tensor_data_dict.keys(): {tensor_data_dict.keys()}')
	for key in tensor_data_dict.keys():
		if type(tensor_data_dict[key]) == torch.Tensor:
			print((key, tensor_data_dict[key].shape))
		else:
			print((key, type(tensor_data_dict[key])))
	
	#here begineth the dashboard

	def visualize_text_feature(feature_index,dataset,llm,crosscoder):
		enc_acts,raw_acts=get_activations(input,llm,crosscoder)
		example_texts=[dataset[i]['text'] for i in range(3)]
		example_activations_encoding=[get_activations(text,llm,crosscoder)[0] for text in example_texts]
		enc_viz_dic={}
		for i in range(len(example_texts)):
			visualization=display_feature_activation_visualization(tokenizer, feature_index=feature_index, example_texts=[example_texts[i]],example_activations=[example_activations_encoding[i]])
			enc_viz_dic[f"Feature: {str(feature_index)}, Text: {str(i)}"]=visualization.data
	
		serve_multiple_visualizations(enc_viz_dic)
		
		

	def dashboard(tensor_data_dict,dataset,llm,crosscoder):
		max_count=tensor_data_dict['max_count']
		max_freqs=max_count.sum(dim=0)
		max_freqs_counts,max_freqs_inds=torch.sort(max_freqs,dim=-1,descending=True)
		visualize_text_feature(max_freqs_inds[300],dataset,llm,crosscoder)
		visualize_text_feature(max_freqs_inds[301],dataset,llm,crosscoder)
		visualize_text_feature(max_freqs_inds[302],dataset,llm,crosscoder)
		#print(f'max_freqs_counts.shape: {max_freqs_counts.shape}')
		#print(f'top feature index {max_freqs_inds[0]}, top feature counts: {max_freqs_counts[0]}')
		#now let's visualiz



	# visualize_feature_stats_from_tensor(tensor_data_dict).show()
	dashboard(tensor_data_dict,dataset,llm,crosscoder)
	exit()
	
	# Save feature statistics to a pickle file for later analysis
	

	#print(f'feature dict keys, shapes {[feature_dict[key].shape for key in feature_dict]}')

	
	
	# old_feat_dict=count_largest_features(dataset,llm,crosscoder,num_samples=1,blocks_to_analyze=[0])
	# visualize_feature_stats(old_feat_dict).show()

	# new_feat_dict=get_feature_stats(dataset,llm,crosscoder,num_samples=10)
	# save_dict(new_feat_dict,'feature_stats_stories_10')

	# visualize_feature_stats_from_tensor(new_feat_dict).show()
	exit()
	# exit()
	# sorted_bnh_vals,sorted_bnh_inds=torch.sort(preacts_PBNH.abs(),dim=-1,descending=True)
	# cumsum_bnh=torch.cumsum(sorted_bnh_vals,dim=-1)
	# max_mean=cumsum_bnh[:,:,0]/cumsum_bnh[:,:,-1]
	# max_mean_mean=max_mean.mean()
	# max_mean_min=max_mean.min()
	# max_mean_max=max_mean.max()
	# max_mean_std=max_mean.std()
	# print(f'max_mean mean: {max_mean_mean}')
	# print(f'max_mean min: {max_mean_min}')
	# print(f'max_mean max: {max_mean_max}')
	# print(f'max_mean std: {max_mean_std}')

	# temp=torch.zeros((preacts_PBNH.shape[1],preacts_PBNH.shape[2]))
	# for batch_ind in range(preacts_PBNH.shape[0]):
	# 	preacts_NH_batch=preacts_PBNH[batch_ind,:,:].abs()
	# 	temp+=get_heatmap_NH(preacts_NH_batch,sort_max=True)
	# temp/=preacts_PBNH.shape[0]


	
	#print(f'mlp_push_through.shape: {mlp_push_through.shape}')

	#Now want to just measure the bias share
	mlp_bias_share_actual=b_ins.abs()/mlp_push_through.abs().mean(dim=1)
	#enc_bias_share=b_dec_XD.abs()/enc_acts.abs().mean(dim=0)
	dec_acts_ln2=torch.stack([dec_acts[:,0,4*block+3,:] for block in range(4)],dim=0)
	dec_bias_mlp=torch.stack([b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
	dec_bias_share_rec=dec_bias_mlp.abs()/dec_acts_ln2.abs().mean(dim=1)

	# wandb_no_penalty = 'biv1u3ig'
	wandb_no_penalty = 'vl9klznb'
	crosscoder_2 = load_crosscoder_from_wandb(
	"dmitry2-uiuc",
	"sleeper-model-diffing",
	wandb_no_penalty,
	"../../.wandb_artifacts",
	DEVICE)

	preacts_PBNH=get_preacts_nocontract(enc_acts,crosscoder_2.W_dec_HXD,crosscoder_2.b_dec_XD,llm,bias=True,block=0)
	print(f'preacts_PBNH.shape: {preacts_PBNH.shape}')
	max_NH=torch.max(preacts_PBNH.abs(),dim=-1)[0]
	print(f'mean max to sum old: {(max_NH.abs()/preacts_PBNH.abs().sum(dim=-1)).mean()}')

	# p_BNH_withbias=get_preacts_nocontract(enc_acts,crosscoder_2.W_dec_HXD,crosscoder_2.b_dec_XD,llm,bias=True,block=0)

	

	# abs_bias_ratio=((nn.GELU()(preacts_PBNH)).sum(dim=-1)-nn.GELU()(p_BNH_withbias).sum(dim=-1))/nn.GELU()(p_BNH_withbias).sum(dim=-1)
	# abs_bias_ratio_mean=abs_bias_ratio.mean()
	# abs_bias_ratio_std=abs_bias_ratio.std()
	# abs_bias_ratio_max=abs_bias_ratio.max()
	# abs_bias_ratio_min=abs_bias_ratio.min()
	# print(f'abs_bias_ratio_mean: {abs_bias_ratio_mean}')
	# print(f'abs_bias_ratio_std: {abs_bias_ratio_std}')
	# print(f'abs_bias_ratio_max: {abs_bias_ratio_max}')
	# print(f'abs_bias_ratio_min: {abs_bias_ratio_min}')

	# exit()
	

	# test_losses=hookpoints_losses(raw_acts,dec_acts,used_names)
	# plot_losses(*test_losses,used_names).show()
	
		
		
	# fig=data_decomposition_heatmap_comparison_datalast(input,llm,crosscoder,crosscoder_2,1)
	# fig.show()
	


	



	def largest_feature_correlation(crosscoder_,llm):
		W_ins=torch.stack([llm.blocks[block].mlp.W_in for block in range(4)],dim=0)
		b_ins=torch.stack([llm.blocks[block].mlp.b_in for block in range(4)],dim=0)
		W_outs=torch.stack([llm.blocks[block].mlp.W_out for block in range(4)],dim=0)
		b_outs=torch.stack([llm.blocks[block].mlp.b_out for block in range(4)],dim=0)
		W_dec_HXD=torch.stack([crosscoder_.W_dec_HXD[:,0,4*block+3,:] for block in range(4)],dim=0)
		b_dec_XD=torch.stack([crosscoder_.b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
		enc_acts,raw_acts=get_activations(input,llm,crosscoder_,tokenizer)
		
		
		
		push_through=push_to_mlp(raw_acts,W_ins,b_ins)
		preacts_PBNH,_=get_neuron_preacts_cutoff(enc_acts,W_dec_HXD,b_dec_XD,W_ins,b_ins,W_outs,b_outs,bias=True)
		# Find the largest absolute values and their indices for each entry
		lead_abs_vals, lead_abs_inds = torch.max(preacts_PBNH.abs(), dim=-1)
		
		# Create indices for gathering the original values with signs
		batch_indices = torch.arange(preacts_PBNH.shape[0])[:, None, None].expand_as(lead_abs_vals)
		seq_indices = torch.arange(preacts_PBNH.shape[1])[None, :, None].expand_as(lead_abs_vals)
		mlp_indices = torch.arange(preacts_PBNH.shape[2])[None, None, :].expand_as(lead_abs_vals)
		
		# Get the original values with signs for the largest absolute values
		lead_features = preacts_PBNH[batch_indices, seq_indices, mlp_indices, lead_abs_inds]
		lead_features = torch.flatten(lead_features)
		
		# Mask out the largest values to find the second largest
		masked_preacts = preacts_PBNH.clone()
		masked_preacts[batch_indices, seq_indices, mlp_indices, lead_abs_inds] = 0
		
		# Find the second largest absolute values and their indices
		second_abs_vals, second_abs_inds = torch.max(masked_preacts.abs(), dim=-1)
		
		# Get the original values with signs for the second largest absolute values
		second_features = masked_preacts[batch_indices, seq_indices, mlp_indices, second_abs_inds]
		second_features = torch.flatten(second_features)
		
		

		
		biases=torch.stack([b_ins for _ in range(preacts_PBNH.shape[1])],dim=1)
		biases=torch.flatten(biases)

		#want the neuron preacts
		print(f'preacts_PBNH.shape: {preacts_PBNH.shape}')
		neuron_preacts=torch.flatten(preacts_PBNH[batch_indices,seq_indices,mlp_indices,:].sum(dim=-1))
		
		

		return lead_features,second_features,biases,neuron_preacts
	


	# lead_features,second_features,biases,neuron_preacts=largest_feature_correlation(crosscoder,llm)


	# samples=100_000
	# random_inds=torch.arange(samples)
	# #random_inds=torch.randperm(lead_features.shape[0])[:samples]
	# fig=make_subplots(rows=3,cols=2)
	
	# fig.add_trace(go.Scatter(x=lead_features[random_inds],y=neuron_preacts[random_inds],mode='markers'),row=1,col=1)
	# fig.add_trace(go.Scatter(x=lead_features[random_inds],y=biases[random_inds],mode='markers'),row=1,col=2)

	
	# fig.add_trace(go.Scatter(x=second_features[random_inds],y=neuron_preacts[random_inds],mode='markers'),row=2,col=1)
	# fig.add_trace(go.Scatter(x=second_features[random_inds],y=biases[random_inds],mode='markers'),row=2,col=2)
	
	# fig.add_trace(go.Scatter(x=lead_features[random_inds],y=second_features[random_inds],mode='markers'),row=3,col=1)
	
	# fig.update_xaxes(title_text="Lead Feature",row=1)
	# fig.update_yaxes(title_text="Neuron Preacts",row=1,col=1)
	# fig.update_yaxes(title_text="Bias",row=1,col=2)
	
	# fig.update_xaxes(title_text="Second Feature",row=2)
	# fig.update_yaxes(title_text="Neuron Preacts",row=2,col=1)
	# fig.update_yaxes(title_text="Bias",row=2,col=2)
	

	# fig.update_yaxes(title_text="Second Feature",row=3,col=1)
	# fig.update_xaxes(title_text="Lead Feature",row=3,col=1)

	# fig.update_layout(title_text="Lead and Second Features (New XC)")
	# fig.update_layout(showlegend=False)

	# fig.show()
	
		

	



	# def get_bias_share(crosscoder_,llm):
	# 	W_ins=torch.stack([llm.blocks[block].mlp.W_in for block in range(4)],dim=0)
	# 	b_ins=torch.stack([llm.blocks[block].mlp.b_in for block in range(4)],dim=0)
	# 	enc_acts,raw_acts=get_activations(input,llm,crosscoder_,tokenizer)
	# 	mlp_push_through=push_to_mlp(raw_acts,W_ins,b_ins)
	# 	#preacts_PBNH=get_preacts_nocontract(enc_acts,crosscoder.W_dec_HXD,b_dec_XD,llm,bias=False,block=0)
	# 	dec_acts=crosscoder_._forward(raw_acts).output_BXD

	# 	dec_acts_ln2=torch.stack([dec_acts[:,0,4*block+3,:] for block in range(4)],dim=0)
	# 	dec_bias_mlp=torch.stack([b_dec_XD[0,4*block+3,:] for block in range(4)],dim=0)
	# 	dec_bias_share_rec=dec_bias_mlp.abs()/dec_acts_ln2.abs().mean(dim=1)

	# 	mlp_bias_share_actual=b_ins.abs()/mlp_push_through.abs().mean(dim=1)
	# 	return mlp_bias_share_actual, dec_bias_share_rec
	
	# #I think you could plot the correlation of the largest two features and the bias?

	# mlp_bias_share_actual_new, dec_bias_share_rec_new=get_bias_share(crosscoder,llm)
	# mlp_bias_share_actual_old, dec_bias_share_rec_old=get_bias_share(crosscoder_2,llm)
		

	# fig=make_subplots(rows=2,cols=2)
	# fig.add_trace(go.Histogram(x=torch.flatten(mlp_bias_share_actual_new).detach().numpy()),row=1,col=1)
	# fig.add_trace(go.Histogram(x=torch.flatten(dec_bias_share_rec_new).detach().numpy()),row=1,col=2)
	# fig.add_trace(go.Histogram(x=torch.flatten(mlp_bias_share_actual_old).detach().numpy()),row=2,col=1)
	# fig.add_trace(go.Histogram(x=torch.flatten(dec_bias_share_rec_old).detach().numpy()),row=2,col=2)
	# fig.update_yaxes(title_text="Count")
	# fig.update_xaxes(title_text="MLP Bias Share of actual MLP acts (same in both XC)",col=1)
	# fig.update_xaxes(title_text="Dec. Bias Share of reconstructed ln2 acts",col=2)
	# fig.update_layout(title_text="Bias Share of MLP and Decoding acts New XC (Top) and Old XC (Bottom)")
	# fig.show()

	#Largest feature correlation with bias

	

	

	

	
	


	



	#test_losses=hookpoints_losses(raw_acts,dec_acts,used_names)
	#plot_losses(*test_losses,used_names).show()

	#data_decomposition_heatmap(enc_acts,llm,W_dec_HXD,b_dec_XD,0).show()
	

	#patched_losses=patched_model_loss(llm,input,crosscoder,used_names)
	#print(f'patched_losses: {patched_losses}')


	
	# wandb_no_penalty = 'rx649y7j'
	# crosscoder_2 = load_crosscoder_from_wandb(
	# "dmitry2-uiuc",
	# "sleeper-model-diffing",
	# wandb_no_penalty,
	# "../../.wandb_artifacts",
	# DEVICE)

	
	# fig=data_decomposition_heatmap_comparison_datalast(input,llm,crosscoder,crosscoder_2,0)
	# fig.update_layout(title_text="Comparison of old (bottom) and new (top) XC")
	# fig.show()



	# mat=make_adjacency_matrix_dominant(llm,enc_acts,crosscoder,eps=1e-6)
	# # print(f"int. metric shape: {mat.shape}")
	# fig=make_subplots(rows=1,cols=1)
	# fixed_colorscale=[[0, 'blue'],[0.5,'white'],[0.51,'green'],[0.52,'white'], [0.9, 'red'], [1, 'black']]
	# fig.add_trace(go.Heatmap(z=mat,colorscale='Viridis'))
	# fig.update_yaxes(title_text="Max Feature",title_font_size=24,tickfont_size=20)
	# fig.update_xaxes(title_text="Top 100 Features",title_font_size=24,tickfont_size=20)
	

	# fig.show()

	#Feature analysis over the dataset

	#Idea 1: Just count the story and token in which each feature is a max?
	features_dic=defaultdict(tuple)

	def get_features_dic(dataset):
		for ind in tqdm(range(1000)):
			story=dataset[ind]["text"]
			enc_acts,raw_acts=get_activations(story,llm,crosscoder,tokenizer)
			for block in range(4):
				W_in = llm.blocks[block].mlp.W_in
				b_in = llm.blocks[block].mlp.b_in
				preacts_BNH=get_preacts_nocontract(enc_acts,crosscoder.W_dec_HXD,crosscoder.b_dec_XD,llm,bias=False,block=block)
				max_inds=torch.argmax(preacts_BNH.abs(),dim=-1)
				uniq_max_inds,counts=torch.unique(max_inds,return_counts=True)
				uniq_mean_value=torch.mean(preacts_BNH[max_inds],dim=0)

				for max_ind,count in zip(uniq_max_inds,counts):
					features_dic[max_ind.item()]+=(ind,count)
		
			print(f'features_dic: {features_dic}')
	
	#test=get_features_dic(dataset)


	#print(f'features_dic: {features_dic}')

	feature_stats_1 = count_largest_features(dataset, llm, crosscoder_2, num_samples=1)

	
	fig_1 = visualize_feature_stats(feature_stats_1)
	fig_1.show()
	exit()
	# Save feature statistics to a pickle file for later analysis

	# Create a timestamp for the filename
	# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	folder_path='/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/data/features'
	# # Save the feature statistics to a pickle file
	# with open(f'{folder_path}/feature_stats_newxc_{timestamp}.pkl', 'wb') as f:
	#     pickle.dump(feature_stats_1, f)
	
	# print(f"Feature statistics saved to feature_stats_newxc_{timestamp}.pkl")

	# fig_1.update_layout(title_text="Max Features by Frequency and Mean Value (New XC)")

	# feature_stats_2 = count_largest_features(dataset, llm, crosscoder_2, num_samples=100)

	# # Save feature statistics to a pickle file for later analysis
	# with open(f'{folder_path}/feature_stats_oldxc_{timestamp}.pkl', 'wb') as f:
	#     pickle.dump(feature_stats_2, f)
	
	# print(f"Feature statistics saved to feature_stats_oldxc_{timestamp}.pkl")

	filename_new='/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/data/features/feature_stats_newxc_20250312_182424.pkl'
	filename_old='/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/data/features/feature_stats_oldxc_20250312_182424.pkl'
	
	data_new= pickle.load(open(filename_new, 'rb'))
	data_old= pickle.load(open(filename_old, 'rb'))

	fig_1=visualize_feature_stats(data_new)
	fig_2=visualize_feature_stats(data_old)
	fig_2.update_layout(title_text="Max Features by Frequency and Mean Value (Old XC)")
	fig_1.update_layout(title_text="Max Features by Frequency and Mean Value (New XC)")
	fig_1.update_yaxes(type='log')
	fig_2.update_yaxes(type='log')
	fig_1.show()
	fig_2.show()

	




