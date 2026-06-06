import torch
from tqdm import tqdm
from einops import rearrange
from typing import Any, List
import numpy as np

def get_activations_batch(
	input_texts: List[str],
	model: Any,                 # transformer‑lens model
	crosscoder: Any             # your cross‑coder
):
	"""
	Process multiple input texts and return their feature activations.
	
	Parameters
	----------
	input_texts : List[str]     # List of text inputs to process
	model : Any                 # transformer‑lens model
	crosscoder : Any            # your cross‑coder
	
	Returns
	--------
	feature_activations_BSH : torch.Tensor   # [batch_size, seq_len (=128), H]
	activations_BSMLD      : torch.Tensor   # [batch_size, seq_len, model, layer, d_model]
	"""
	batch_size = len(input_texts)
	device = model.cfg.device
	
	# ------------------------------------------------------------------
	# 1. Tokenize all inputs and pad/truncate to fixed length (128)
	# ------------------------------------------------------------------
	all_tokens = []
	seq_len = 128
	
	for text in input_texts:
		tokens = model.tokenizer.encode(text)[:seq_len]
		# Pad if necessary (though we'll typically truncate)
		if len(tokens) < seq_len:
			tokens = tokens + [0] * (seq_len - len(tokens))
		all_tokens.append(tokens)
	
	# Stack into a batch tensor
	tokens_batch = torch.tensor(all_tokens, device=device)
	
	# ------------------------------------------------------------------
	# 2. Cache only the hookpoints we need
	# ------------------------------------------------------------------
	hook_names = [
		"blocks.0.hook_resid_pre",  "blocks.0.ln1.hook_normalized",
		"blocks.0.hook_resid_mid",  "blocks.0.ln2.hook_normalized",
		"blocks.1.hook_resid_pre",  "blocks.1.ln1.hook_normalized",
		"blocks.1.hook_resid_mid",  "blocks.1.ln2.hook_normalized",
		"blocks.2.hook_resid_pre",  "blocks.2.ln1.hook_normalized",
		"blocks.2.hook_resid_mid",  "blocks.2.ln2.hook_normalized",
		"blocks.3.hook_resid_pre",  "blocks.3.ln1.hook_normalized",
		"blocks.3.hook_resid_mid",  "blocks.3.ln2.hook_normalized",
		"blocks.3.hook_resid_post"
	]
	
	with torch.no_grad():
		_, cache = model.run_with_cache(tokens_batch, names_filter=hook_names)
	
	# ------------------------------------------------------------------
	# 3. Stack the cached tensors and move to CPU to free GPU memory
	# ------------------------------------------------------------------
	activations_BSLD = torch.stack(
		[cache[name] for name in hook_names], dim=2
	).cpu()
	
	del cache
	torch.cuda.empty_cache()
	
	# ------------------------------------------------------------------
	# 4. Rearrange dimensions to the expected format
	# ------------------------------------------------------------------
	activations_BSMLD = activations_BSLD.unsqueeze(2)  # add model dimension [B, S, 1, L, D]
	B, S, M, L, D = activations_BSMLD.shape # Get dimensions

	# ------------------------------------------------------------------
	# 5. Process with the cross-coder in batches if needed
	# ------------------------------------------------------------------
	crosscoder_device = crosscoder.W_dec_HXD.device
	
	with torch.no_grad():
		# Move to crosscoder's device for encoding
		activations_device = activations_BSMLD.to(crosscoder_device, non_blocking=True)
		
		# <<< START FIX: Reshape for _encode_BH and reshape output >>>
		# Reshape input to combine Batch and Sequence dimensions: [(B*S), M, L, D]
		activations_reshaped = rearrange(activations_device, 'b s m l d -> (b s) m l d')

		# Process batch through crosscoder, expecting [(B*S), H]
		feature_activations_flat = crosscoder._encode_BH(activations_reshaped).cpu()

		# Reshape output back to [B, S, H]
		# Check if output is 1D (e.g., if B*S*H = H), handle potential squeeze case
		if feature_activations_flat.ndim == 1 and B*S == 1:
			 H = feature_activations_flat.shape[0]
			 feature_activations_BSH = feature_activations_flat.reshape(B, S, H)
		elif feature_activations_flat.ndim == 2: 
			 # Assuming shape is [(B*S), H]
			 _, H = feature_activations_flat.shape
			 feature_activations_BSH = rearrange(feature_activations_flat, '(b s) h -> b s h', b=B, s=S)
		else:
			# Fallback or error handling if shape is unexpected
			print(f"Warning: Unexpected output shape from _encode_BH: {feature_activations_flat.shape}")
			feature_activations_BSH = feature_activations_flat # Or raise error
		# <<< END FIX >>>

	torch.cuda.empty_cache()
	
	return feature_activations_BSH, activations_BSMLD

def get_preacts_nocontract(enc_acts:torch.Tensor, W_dec_HMLD:torch.Tensor, b_dec_MLD:torch.Tensor,llm:object,block:int=0, bias=True):
	"""
	Get the preacts without contracting the feature dimension
	"""
	# --- Determine target device from llm ---
	W_in = llm.blocks[block].mlp.W_in
	b_in = llm.blocks[block].mlp.b_in
	device = W_in.device # Assume llm's weights define the target device

	# --- Ensure all input tensors are on the target device ---
	enc_acts = enc_acts.to(device)
	W_dec_HMLD = W_dec_HMLD.to(device)
	b_dec_MLD = b_dec_MLD.to(device)
	# W_in and b_in are already on the device from llm

	#Note that you have to divide by the number of features to get the correct bias
	if bias:
		bias_factor = 1
	else:
		bias_factor = 0
	hidden_dim = enc_acts.shape[1]
	# device = enc_acts.device  # Get the device of enc_acts # <<< REMOVED: Now derived from W_in
	# W_dec_HMLD = W_dec_HMLD.to(device) # Move W_dec_HMLD to the correct device # <<< MOVED UP
	# b_dec_MLD = b_dec_MLD.to(device)   # Move b_dec_MLD to the correct device # <<< MOVED UP
	dec_nocontract = (enc_acts[:,:,None] * W_dec_HMLD[None,:,0,4*block+3,:]) + bias_factor*b_dec_MLD[0,4*block+3,:].unsqueeze(0)/hidden_dim


	#OK so decoding is correct, then let's push through the mlp
	# W_in = W_in.to(device) # Move W_in to the correct device # <<< REMOVED: Already on device
	pre_acts=einsum(W_in, dec_nocontract, "d_model d_mlp, batch hidden d_model -> batch hidden d_mlp")
	# b_in = b_in.to(device) # Move b_in to the correct device # <<< REMOVED: Already on device
	pre_acts += bias_factor*b_in/hidden_dim
	pre_acts = rearrange(pre_acts, 'batch hidden d_mlp -> batch d_mlp hidden')

	return pre_acts
