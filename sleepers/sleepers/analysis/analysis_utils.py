from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser
import pickle
from datetime import datetime
import os
import torch
from einops import rearrange,einsum
from typing import Any
import numpy as np
from tqdm import tqdm
import einops
import torch.nn as nn



def get_activations(
	input_text: str,
	model: Any,                 # transformer‑lens model
	crosscoder: Any             # your cross‑coder
):
	"""
	Returns
	--------
	feature_activations_SH : torch.Tensor   # [seq_len  (=128) ,  H]
	activations_SMLD      : torch.Tensor   # [(B·S) , model , layer , d_model]
	"""
	# ------------------------------------------------------------------
	# 1.  Tokenise and keep the tensor on the *same* device as the model
	# ------------------------------------------------------------------
	tokens = torch.tensor(
		model.tokenizer.encode(input_text)[:128],
		device=model.cfg.device  # usually "cuda" if you created the model there
	)

	# ------------------------------------------------------------------
	# 2.  Cache only the hookpoints you really need
	#     Wrap everything in no‑grad to avoid autograd buffers.
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
		_, cache = model.run_with_cache(tokens.unsqueeze(0), names_filter=hook_names)

	# ------------------------------------------------------------------
	# 3.  Stack the cached tensors, then *immediately* drop the originals
	# ------------------------------------------------------------------
	activations_BSLD = torch.stack(
		[cache[name] for name in hook_names], dim=2
	).cpu()                      # move to CPU to free GPU VRAM

	del cache                    # <<< this is enough
	torch.cuda.empty_cache()     # release GPU memory


	# ------------------------------------------------------------------
	# 4.  Rearrange exactly as before
	# ------------------------------------------------------------------
	activations_BSMLD = activations_BSLD.unsqueeze(2)                 # add model‑dim
	activations_SMLD  = rearrange(activations_BSMLD, "b s m l d -> (b s) m l d")

	# ------------------------------------------------------------------
	# 5.  Encode with the cross‑coder
	#     (cross‑coder expects activations on the same device as itself)
	# ------------------------------------------------------------------
	with torch.no_grad():
		feature_activations_SH = crosscoder._encode_BH(
			activations_SMLD.to(crosscoder.W_dec_HXD.device, non_blocking=True)
		).cpu()                  # bring result back to CPU; free GPU right away

	torch.cuda.empty_cache()     # tidy up after cross‑coder

	return feature_activations_SH, activations_SMLD

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
	
	# Choose hookpoint index based on architecture
	# GPT-2 pattern: [resid_pre, ln1, resid_mid, ln2] -> MLP input at index 3
	# Pythia pattern: [resid_pre, ln1, ln2, resid_post] -> MLP input at index 2
	mlp_hookpoint_idx = 4*block + (2 if pythia_format else 3)
	
	dec_nocontract = (enc_acts[:,:,None] * W_dec_HMLD[None,:,0,mlp_hookpoint_idx,:]) + bias_factor*b_dec_MLD[0,mlp_hookpoint_idx,:].unsqueeze(0)/hidden_dim

	#OK so decoding is correct, then let's push through the mlp
	pre_acts=einsum(W_in, dec_nocontract, "d_model d_mlp, batch hidden d_model -> batch hidden d_mlp")
	pre_acts += bias_factor*b_in/hidden_dim
	pre_acts = rearrange(pre_acts, 'batch hidden d_mlp -> batch d_mlp hidden')

	return pre_acts

def get_preacts_nocontract_faster(
	enc_acts: torch.Tensor,
	W_dec_HMLD: torch.Tensor,
	b_dec_MLD: torch.Tensor,
	llm: object,
	block: int = 0,
	bias: bool = True,
	pythia_format = None,
) -> torch.Tensor:
	"""
	Faster sparse version: fuse decoder & input weights, then only touch active features.
	
	Args:
		pythia_format: If True, use Pythia hookpoint indexing (MLP input at index 2).
		               If False, use GPT-2 hookpoint indexing (MLP input at index 3).
	"""
	# Auto-detect architecture if not specified
	if pythia_format is None:
		model_name = getattr(llm.cfg, 'model_name', '')
		pythia_format = 'pythia' in model_name.lower()
	
	# --- 1) grab shapes & hooks ---
	B, H = enc_acts.shape           # batch, hidden
	d_model, d_mlp = llm.cfg.d_model, llm.blocks[block].mlp.b_in.shape[0]
	hook = 4 * block + (2 if pythia_format else 3)
	hidden_dim = H

	# --- 2) pull out W_in and b_in from the LLM ---
	W_in = llm.blocks[block].mlp.W_in           # [d_model, d_mlp]
	b_in = llm.blocks[block].mlp.b_in           # [d_mlp]

	# --- 3) fuse decoder & input weights once ---
	#    W_dec_HMLD: [hidden, num_models, num_hooks, d_model]
	W_dec_hook = W_dec_HMLD[:, 0, hook, :]      # [hidden, d_model]
	W_comp = W_dec_hook @ W_in                  # [hidden, d_mlp]

	# --- 4) allocate output and only fill in nonzero positions ---
	pre_acts = enc_acts.new_zeros((B, H, d_mlp))
	batch_idx, hidden_idx = (enc_acts != 0).nonzero(as_tuple=True)
	vals = enc_acts[batch_idx, hidden_idx].unsqueeze(-1) * W_comp[hidden_idx]  # [nnz, d_mlp]
	pre_acts[batch_idx, hidden_idx] = vals

	# --- 5) add combined bias ---
	if bias:
		# original code did: dec_bias -> W_in -> + b_in
		dec_bias = b_dec_MLD[0, hook]            # [d_model]
		# composite_bias[i] = sum_j W_in[j,i] * dec_bias[j]  +  b_in[i]
		composite_bias = (dec_bias @ W_in + b_in) / hidden_dim   # [d_mlp]
		pre_acts += composite_bias.view(1, 1, -1)

	# --- 6) reorder to match old output shape ---
	pre_acts = rearrange(pre_acts, 'batch hidden d_mlp -> batch d_mlp hidden')
	return pre_acts





def get_preacts_nocontract_all(enc_acts:torch.Tensor, crosscoder:object,llm:object, bias=True):
	"""
	Get the preacts without contracting the feature dimension for all blocks
	"""
	blocks=range(len(llm.blocks))
	W_ins=torch.stack([llm.blocks[block].mlp.W_in for block in blocks])
	b_ins=torch.stack([llm.blocks[block].mlp.b_in for block in blocks])


	W_dec_PHD=torch.stack([crosscoder.W_dec_HXD[:,0,4*block+3,:] for block in blocks])
	b_dec_PD=torch.stack([crosscoder.b_dec_XD[0,4*block+3,:] for block in blocks])

	hidden_dim=enc_acts.shape[-1]
	dec_nocontract = (enc_acts[None,:,:,None] * W_dec_PHD[:,None,:,:])+ bias*b_dec_PD.unsqueeze(1).unsqueeze(2)/hidden_dim
	
	
	pre_acts=einsum(W_ins, dec_nocontract, "block d_model d_mlp, block batch hidden d_model -> block batch hidden d_mlp")
	
	
	pre_acts += bias*b_ins[:,None,None,:,]/hidden_dim
	pre_acts = rearrange(pre_acts, 'block batch hidden d_mlp -> block batch d_mlp hidden')


	return pre_acts

def get_preacts_mlp(input_texts,llm,crosscoder,block=1):
	"""
	Analog of get_activations, but this is for analyzing the pre-activations of the MLP,
	rather than the activations of the features.
	First, I want to just see the features at the ML at each token, averaged over the (abs) of the neurons.
	"""

	enc_acts,resid_acts=get_activations(input_texts,llm,crosscoder)
	if block is None:
		preacts_mlp=torch.stack([get_preacts_nocontract(enc_acts,crosscoder.W_dec_HXD,crosscoder.b_dec_XD,llm,bias=True,block=block) for block in range(4)],dim=0)
	else:
		preacts_mlp=get_preacts_nocontract(enc_acts,crosscoder.W_dec_HXD,crosscoder.b_dec_XD,llm,bias=True,block=block)
		preacts_mlp=preacts_mlp.unsqueeze(0)
	
	preacts_PBH=preacts_mlp.abs().mean(dim=2)
	
	return preacts_PBH[0]
	


def feature_interactions_mlp(input_text,llm,crosscoder,dataset=None,block=0,num_datapoints=1):
	num_features = crosscoder.W_dec_HXD.shape[0]
	num_tokens = 128

	W_in = llm.blocks[block].mlp.W_in
	b_in = llm.blocks[block].mlp.b_in
	W_out = llm.blocks[block].mlp.W_out
	W_dec = crosscoder.W_dec_HXD[:,0,4*block+3,:]
	b_dec = crosscoder.b_dec_XD[0,4*block+3,:]
	W_dec_W_in = W_dec @ W_in

	device = W_dec_W_in.device # Get the target device from W_dec_W_in
	normalisation = W_out.norm(dim=1).to(device) # Ensure normalisation is also on the correct device
	
	feature_interactions = torch.zeros((num_tokens,num_features, num_features), device=device)

	if input_text is None:
		pass
	else:
		num_datapoints=1

	for i in range(num_datapoints):
		# Determine text for this datapoint
		if input_text is None:
			text_i = dataset[i]["text"]
		else:
			text_i = input_text

		# Obtain activations once and move to target device
		feature_activations_SH, _ = get_activations(text_i, llm, crosscoder)
		acts_SH = feature_activations_SH.to(device)
		feature_activations_SH=feature_activations_SH.to(device)

		# Compute and accumulate dot-product matrix and squared norms
		feats_map = einops.einsum(acts_SH, acts_SH, "batch hidden1, batch hidden2 -> hidden1 hidden2")
		feats_dp_nm = feats_map
		feats_squares = feats_map.diagonal()

		seq_len=feature_activations_SH.shape[0]
		for s in range(seq_len):#128
			active_features = torch.where(feature_activations_SH[s, :] != 0.0)[0]
			per_feature_preactivations_HA = W_dec_W_in[active_features, :] * feature_activations_SH[s, active_features, None]
			per_feature_abs_preactivations_HA = per_feature_preactivations_HA.abs()
			max_features = active_features[torch.argmax(per_feature_abs_preactivations_HA, dim=0)]
			rows = max_features
			cols = active_features
			# use np.ix_ to broadcast the two 1d index arrays into a 2d block
			interaction_HA = per_feature_abs_preactivations_HA*normalisation[None, :]
			feature_interactions[s,rows.unsqueeze(1), cols] += interaction_HA.T  # TODO do normalisation
			feature_interactions[s,rows, rows] = 0
		

	
	feature_interactions = feature_interactions / (num_datapoints)
	return feature_interactions

def cosine_sim_ints(input_text,llm,crosscoder,dataset=None,block=0,num_datapoints=1):
	num_features = crosscoder.W_dec_HXD.shape[0]
	num_tokens = 128

	W_in = llm.blocks[block].mlp.W_in
	b_in = llm.blocks[block].mlp.b_in
	W_out = llm.blocks[block].mlp.W_out
	W_dec = crosscoder.W_dec_HXD[:,0,4*block+3,:]
	b_dec = crosscoder.b_dec_XD[0,4*block+3,:]
	W_dec_W_in = W_dec @ W_in

	device = W_dec_W_in.device # Get the target device from W_dec_W_in
	normalisation = W_out.norm(dim=1).to(device) # Ensure normalisation is also on the correct device
	
	feature_interactions = torch.zeros((num_tokens,num_features, num_features), device=device)

	if input_text is None:
		pass
	else:
		num_datapoints=1

	feats_dp_nm=torch.zeros((num_features,num_features),device=device)
	feats_squares=torch.zeros((num_features),device=device)
	for i in range(num_datapoints):
		# Determine text for this datapoint
		if input_text is None:
			text_i = dataset[i]["text"]
		else:
			input = input_text
		
		#feature_activations_SH,activations_SMLD = get_activations(input, llm,crosscoder)
		#feature_activations_SH = feature_activations_SH.to(device) # Move activations to the target device
		#seq_len=feature_activations_SH.shape[0]
		acts_SH=get_activations(input,llm,crosscoder)[0].to(device)
		feats_map=einops.einsum(acts_SH,acts_SH,"batch hidden1,batch hidden2 -> hidden1 hidden2")
		feats_dp_nm+=feats_map.to(device)
		feats_squares+=feats_map.diagonal()
	#feats_dp_nm/=num_datapoints
	#feats_squares/=num_datapoints
	
	#feats_cosine=feats_dp_nm/torch.sqrt(feats_squares[:,None]*feats_squares[None,:])

	return feats_dp_nm,feats_squares
	
			


# Have factored out feature_interactions_mlp_sequence to avoid repeating code but significantly faster
# if you inline it as commented out, or could probably cache some things

# Return an array of feature interactions averaged across neurons and sequence positions, dimensions (num crosscoder features, num crosscoder features)
def feature_interactions_sum(layer, num_datapoints,dataset,llm,crosscoder,num_features=1536):
	# W_in = llm.blocks[layer].mlp.W_in
	# W_out = llm.blocks[layer].mlp.W_out
	# W_dec = crosscoder.W_dec_HXD[:,0,hookpoints.index(f"blocks.{layer}.ln2.hook_normalized"),:]
	# W_dec_W_in = W_dec @ W_in
	
	# normalisation = W_out.norm(dim=1)
	
	feature_interactions = np.zeros((num_features, num_features))
	for i in range(num_datapoints):
		input = dataset[i]["text"]
		feature_interactions_sequence = feature_interactions_mlp(input, llm,crosscoder,block=layer).detach().cpu().numpy()
		feature_interactions += feature_interactions_sequence.sum(axis=0)
		# activations_SMLD = get_llm_activations_SMPD(input, llm)
		# feature_activations_SH = crosscoder._encode_BH(activations_SMLD)
		# for s in range(128):
		#     active_features = torch.where(feature_activations_SH[s, :] != 0.0)[0]
		#     per_feature_preactivations_HA = W_dec_W_in[active_features, :] * feature_activations_SH[s, active_features, None]
		#     per_feature_abs_preactivations_HA = per_feature_preactivations_HA.abs()
		#     max_features = active_features[torch.argmax(per_feature_abs_preactivations_HA, dim=0)]
		#     rows = max_features.cpu().numpy()
		#     cols = active_features.cpu().numpy()
		#     # use np.ix_ to broadcast the two 1d index arrays into a 2d block
		#     interaction_HA = per_feature_abs_preactivations_HA*normalisation[None, :]
		#     feature_interactions[np.ix_(rows, cols)] += interaction_HA.T.cpu().numpy()
		#     feature_interactions[rows, rows] = 0
	
	feature_interactions = feature_interactions / (num_datapoints * 128)
	return feature_interactions

def feature_interactions_alltokens(layer, num_datapoints,dataset,llm,crosscoder,num_features=1536):
	# W_in = llm.blocks[layer].mlp.W_in
	# W_out = llm.blocks[layer].mlp.W_out
	# W_dec = crosscoder.W_dec_HXD[:,0,hookpoints.index(f"blocks.{layer}.ln2.hook_normalized"),:]
	# W_dec_W_in = W_dec @ W_in
	
	# normalisation = W_out.norm(dim=1)
	
	feature_interactions = np.zeros((num_datapoints,num_features, num_features))
	for i in tqdm(range(num_datapoints)):
		input = dataset[i]["text"]
		feature_interactions_sequence = feature_interactions_mlp(input, llm,crosscoder,block=layer).detach().cpu().numpy()
		feature_interactions[i] = feature_interactions_sequence.mean(axis=0)
		# activations_SMLD = get_llm_activations_SMPD(input, llm)
		# feature_activations_SH = crosscoder._encode_BH(activations_SMLD)
		# for s in range(128):
		#     active_features = torch.where(feature_activations_SH[s, :] != 0.0)[0]
		#     per_feature_preactivations_HA = W_dec_W_in[active_features, :] * feature_activations_SH[s, active_features, None]
		#     per_feature_abs_preactivations_HA = per_feature_preactivations_HA.abs()
		#     max_features = active_features[torch.argmax(per_feature_abs_preactivations_HA, dim=0)]
		#     rows = max_features.cpu().numpy()
		#     cols = active_features.cpu().numpy()
		#     # use np.ix_ to broadcast the two 1d index arrays into a 2d block
		#     interaction_HA = per_feature_abs_preactivations_HA*normalisation[None, :]
		#     feature_interactions[np.ix_(rows, cols)] += interaction_HA.T.cpu().numpy()
		#     feature_interactions[rows, rows] = 0
	return feature_interactions


def save_dict(data_dict, filename, path=None):

	# Create a timestamp for the filename
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	
	# Define folder path for saving features
	if path is None:
		folder_path = './data/features'
	else:
		folder_path = path
	
	# Create directory if it doesn't exist
	os.makedirs(folder_path, exist_ok=True)
	
	end_path=f'{folder_path}/{filename}_{timestamp}.pkl'
	# Save the feature statistics to a pickle file
	with open(end_path, 'wb') as f:
		pickle.dump(data_dict, f)
	
	print(f"Feature statistics saved to {end_path}")

	return data_dict,end_path

def load_dict(filepath):
	with open(filepath, 'rb') as f:
		data_dict = pickle.load(f)
	return data_dict


def serve_multiple_visualizations(visualizations_dict, start_port=8000):
	"""
	Args:
		visualizations_dict: A dictionary where keys are titles/descriptions and values are HTML content
		start_port: The starting port number to use for the servers
	"""
	servers = []
	threads = []
	
	for i, (title, html_content) in enumerate(visualizations_dict.items()):
		port = start_port + i
		
		# Create HTML for this visualization
		html = f"""
		<html>
		<head>
			<title>{title}</title>
			<style>
				body {{ font-family: Arial, sans-serif; margin: 20px; }}
				.visualization-container {{ margin-bottom: 40px; }}
			</style>
		</head>
		<body>
			<div class="visualization-container">
				<h2>{title}</h2>
				{html_content}
			</div>
		</body>
		</html>
		"""

		# Create custom handler with our HTML content
		class Handler(SimpleHTTPRequestHandler):
			def do_GET(self):
				self.send_response(200)
				self.send_header('Content-type', 'text/html')
				self.end_headers()
				self.wfile.write(html.encode())

		# Start server in a separate thread
		server = HTTPServer(('localhost', port), Handler)
		thread = threading.Thread(target=server.serve_forever)
		thread.daemon = True
		thread.start()
		
		# Store server and thread
		servers.append(server)
		threads.append(thread)
		
		# Open in browser
		webbrowser.open(f'http://localhost:{port}')
		
		print(f"Visualization '{title}' running at http://localhost:{port}")
	
	print("\nPress Ctrl+C to stop all servers")
	
	try:
		# Wait for keyboard interrupt
		while True:
			input()
	except KeyboardInterrupt:
		# Shutdown all servers
		for server in servers:
			server.shutdown()
		print("\nAll servers stopped")

def propagate_preacts(preacts,resid_mid,W_out,b_out,block):
	mlp_post=nn.GELU()(preacts)
	#print(f'mlp_post shape: {mlp_post.shape}')
	#print(f'W_out shape: {W_out.shape}')
	if len(mlp_post.shape)==3:
		mlp_post=mlp_post.squeeze(0)
	mlp_out=einops.einsum(W_out,mlp_post,"d_mlp d_model, batch d_mlp -> batch d_model")+b_out
	resid_post=resid_mid+mlp_out
	return resid_post