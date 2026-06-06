import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from IPython.display import HTML
from textwrap import dedent
import tempfile
import pathlib
import webbrowser
from datetime import datetime
import os
import sys
import pickle
import time
from tqdm import tqdm
import einops
from datasets import load_dataset
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.analysis.ft_analysis_util import display_feature_activation_visualization
from sleepers.analysis.analysis_utils import (
	save_dict, 
	load_dict, 
	get_preacts_mlp, 
	get_activations, 
	feature_interactions_mlp, 
	get_preacts_nocontract_faster,
	get_preacts_nocontract,
	feature_interactions_sum,
	feature_interactions_alltokens,
	cosine_sim_ints
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


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




def get_feature_interactions(crosscoder,llm,dataset,num_samples,save_dir=None):
	#num_samples = 100
	feat_ints=torch.zeros((4,1536,1536),device=DEVICE)
	feats_ints_nonzero_average=torch.zeros((4,1536,1536),device=DEVICE)
	#tensor_list=[]
	layer_tensor=torch.zeros((4,1536,1536),device=DEVICE)
	with torch.no_grad():
		for story_idx in tqdm(range(num_samples)):
			feat_ints_onestory=torch.zeros((1536,1536),device=DEVICE)
			for layer in range(4):
				input_text=dataset[story_idx]['text']
				interactions_tensor=feature_interactions_mlp(input_text,llm,crosscoder,layer)
				
				
				
				token_summed_tensor=interactions_tensor.sum(dim=0)

				nonzero_counts=(interactions_tensor>0).sum(dim=0)
				feats_ints_nonzero_average[layer] += (interactions_tensor.sum(dim=0) / nonzero_counts.clamp(min=1))# * (nonzero_counts > 0)
				feat_ints_onestory+=token_summed_tensor
				layer_tensor[layer]+=token_summed_tensor
			#tensor_list.append(feat_ints_onestory.detach().cpu())
		feat_ints/=num_samples
		layer_tensor/=num_samples
		feats_ints_nonzero_average/=num_samples
		#feat_ints_perstory = torch.stack(tensor_list)

		if save_dir is None:
			return feat_ints, layer_tensor, feats_ints_nonzero_average
		else:
			os.makedirs(save_dir, exist_ok=True)
			pickle_path = f"{save_dir}/feat_ints_{crosscoder_name}_samples_{num_samples}.pkl"
			with open(pickle_path, "wb") as f:
				pickle.dump(feat_ints, f)
			
			print(f"Large tensor saved to {pickle_path}")
			
			summed_path=f"{save_dir}/summed_feat_ints_{crosscoder_name}_samples_{num_samples}.pkl"
			with open(summed_path, "wb") as f:
				pickle.dump(layer_tensor, f)
			
			summed_path_nonzero=f"{save_dir}/summed_feat_ints_nonzero_{crosscoder_name}_samples_{num_samples}.pkl"
			with open(summed_path_nonzero, "wb") as f:
				pickle.dump(feats_ints_nonzero_average, f)
			
			return feat_ints, layer_tensor, feats_ints_nonzero_average

def get_feature_interactions_cosine(crosscoder,llm,dataset,num_samples,save_dir=None):
	cosine_ints=torch.zeros((1536,1536),device=DEVICE)
	
	#tensor_list=[]
	with torch.no_grad():
		for story_idx in tqdm(range(num_samples)):
			feat_ints_onestory=torch.zeros((1536,1536),device=DEVICE)
			for layer in range(4):
				input_text=dataset[story_idx]['text']
				cosine_ints_FF,_=cosine_sim_ints(input_text,llm,crosscoder,layer)
				
				cosine_ints+=cosine_ints_FF
				
			#tensor_list.append(feat_ints_onestory.detach().cpu())
		
		#feat_ints_perstory = torch.stack(tensor_list)
		cosine_ints=cosine_ints/(torch.sqrt(torch.diag(cosine_ints)[:,None]*torch.diag(cosine_ints)[None,:])+1e-12)
		cosine_ints[torch.arange(1536),torch.arange(1536)]=0
		if save_dir is None:
			return cosine_ints
		else:
			os.makedirs(save_dir, exist_ok=True)
			pickle_path = f"{save_dir}/cosine_ints_{crosscoder_name}_samples_{num_samples}.pkl"
			with open(pickle_path, "wb") as f:
				pickle.dump(cosine_ints, f)
			
			print(f"cosine ints tensor saved to {pickle_path}")
			
			
			return cosine_ints

def make_ints_table(crosscoder,llm,dataset,num_samples):
	#feat_ints, layer_tensor, feats_ints_nonzero_average=get_feature_interactions(crosscoder,llm,dataset,num_samples)
	feats_ints_nonzero_average=get_feature_interactions_cosine(crosscoder,llm,dataset,num_samples)
	#print(f'feat_ints.shape: {feat_ints.shape}, layer_tensor.shape: {layer_tensor.shape}, feats_ints_nonzero_average.shape: {feats_ints_nonzero_average.shape}')
	#raise Exception('Stop here')
	#print(f'shape feats_ints_nonzero_average: {feats_ints_nonzero_average.shape}')
	#sys.exit()
	#feat_ints=torch.mean(feats_ints_nonzero_average,dim=0).detach().cpu().numpy()
	#feat_ints=feat_ints.mean(axis=0)/3072 #I think you never divided by the number of neurons
	#symmetrize
	#feat_ints=(feats_ints+feats_ints.T)/2
	feat_ints=feats_ints_nonzero_average.detach().cpu().numpy()
	
	#token_feat_path=f"feature_interactions/token_feats_abs_nonzeromean_{crosscoder_name}_samples_{100}.pkl"
	#token_feats_VH=pickle.load(open(token_feat_path,'rb'))
	
	
	#topk_tokens=torch.topk(token_feats_VH,k=5,dim=0).indices
	
	
	
	
	flat_feat_ints = feat_ints.flatten()
	top_indices = np.argsort(flat_feat_ints)[::-1][:40][::2]  # Sort in descending order and get top 20, 2 because of symmetry

	

	
	# Get the corresponding feature pairs (i, j) for these indices
	top_feature_pairs = np.unravel_index(top_indices, feat_ints.shape)
	
	# Create a list of (feature_i, feature_j, interaction_value) tuples
	top_interactions = [(top_feature_pairs[0][i], top_feature_pairs[1][i], flat_feat_ints[top_indices[i]]) 
						for i in range(len(top_indices))]

	for i, (feat_i, feat_j, interaction) in enumerate(top_interactions):
		print(f"Interaction {i+1}:")
				
		
		# Get explanation for feature i, handling case where no explanation is found
		feat_i_explanation = "No explanation found"
		if len(df.loc[df['feature_id'] == feat_i, 'explanation'].values) > 0:
			feat_i_explanation = df.loc[df['feature_id'] == feat_i, 'explanation'].values[0]
		
		# Get explanation for feature j, handling case where no explanation is found
		feat_j_explanation = "No explanation found"
		if len(df.loc[df['feature_id'] == feat_j, 'explanation'].values) > 0:
			feat_j_explanation = df.loc[df['feature_id'] == feat_j, 'explanation'].values[0]
		
		print(f"  Feature {feat_i}: Explanation: {feat_i_explanation}")#Top tokens: {llm.tokenizer.decode(topk_tokens[:,feat_i])}
		print(f"  Feature {feat_j}: Explanation: {feat_j_explanation}")# Top tokens: {llm.tokenizer.decode(topk_tokens[:,feat_j])}")
		print(f"  Interaction value: {interaction}")


if __name__ == "__main__":
	import pandas as pd
	crosscoder_name = "86u64trx"
	# Path to your CSV file – adjust as needed
	csv_path = f"/root/crosscoders-feature-interactions/sleepers/sleepers/autointerp/autointerp_data/explanations_{crosscoder_name}_nohate.csv"
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	# Read the CSV into a pandas DataFrame
	df = pd.read_csv(csv_path)

	# Simple sanity check
	print(f"Loaded DataFrame with shape: {df.shape}")
	print(df.head())

	print(df["feature_id"][0])

	dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")

	dataset = dataset.filter(lambda x: x['is_training'] == True)

	llm = build_llm_lora(
		base_model_repo="roneneldan/TinyStories-Instruct-33M",
		lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
		cache_dir=None,
		device=DEVICE,
		dtype=None,
	)

	wandb_run_name = "1k68kpv5"  # example – adjust as needed, base XC, l=1000
	#wandb_run_name='ckubmeg1' #l=1000, bias=True, DF XC
	wandb_run_name_unpenalized='86u64trx' #l=0, bias=True, base XC
	#wandb_run_name='v7128kc4' #l=1000, mlp_bias=True, DF XC (for sure)

	crosscoder = load_crosscoder_from_wandb(
		"dmitry2-uiuc", "sleeper-model-diffing", crosscoder_name, "../../.wandb_artifacts", DEVICE
	)

	# crosscoder_unpenalized = load_crosscoder_from_wandb(
	#     "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name_unpenalized, "../../.wandb_artifacts", DEVICE
	# )

	#get_token_counts(dataset,llm,crosscoder)
	# tot_abs, counts = get_token_counts(dataset,llm,crosscoder,num_datapoints=300)

	# Calculate feature interactions


	
	make_ints_table(crosscoder,llm,dataset,100)

	# Save feature interactions to a pickle file
	# import pickle
	# import os
	
	# # Create directory if it doesn't exist
   

	# sys.exit()

	#crosscoder_name="86u64trx"

	#cosine_ints=get_feature_interactions_cosine(crosscoder,llm,dataset,100)
	
	# Find the top 20 feature interactions
	# pickle_path = f"feature_interactions/summed_feat_ints_nonzero_{crosscoder_name}_samples_{100}.pkl"
	# with open(pickle_path, "rb") as f:
	# 	feat_ints = pickle.load(f)

		