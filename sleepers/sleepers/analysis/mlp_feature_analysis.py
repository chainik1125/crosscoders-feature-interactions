import numpy as np
from matplotlib import pyplot as plt
import sys
from pathlib import Path
import torch
import pandas as pd
from sleepers.scripts.utils import load_crosscoder_from_wandb
import wandb

from sleepers.scripts.llms import build_llm_lora
from sleepers.analysis.ft_analysis_util import display_feature_activation_visualization
from sleepers.analysis.mlp_analysis import get_activations,get_preacts_nocontract





def make_feat_table(llm:object,enc_acts_BH:torch.Tensor, tokens:any,crosscoder:object, block:int=0,eps:float=1e-6) -> pd.DataFrame:
	
	preacts_BNH=get_preacts_nocontract(enc_acts_BH,crosscoder.W_dec_HXD,crosscoder.b_dec_XD,llm,bias=False,block=0)

	print(f'preacts_BNH.shape: {preacts_BNH.shape}')
	
	
	avg_activation_ratios = (preacts_BNH.abs()/preacts_BNH.abs().sum(dim=-1,keepdim=True)).mean(dim=0)
	sorted_values, top_feature_indices = torch.sort(avg_activation_ratios, dim=-1, descending=True)

	# Number of neurons
	num_neurons = top_feature_indices.shape[0]

	# Initialize list to store table rows
	table_rows = []

	for neuron in range(num_neurons):
		# Get top 5 feature indices for this neuron
		top_feats = top_feature_indices[neuron, :5].tolist()

		# Get activation values for top 5 features across all dataset entries
		# signed_sum=preacts_BNH[:,neuron,:].mean(dim=0)
		# abs_sum=preacts_BNH[:,neuron,:].abs().mean(dim=0)
		
		# top_acts_ratio_signed=((signed_sum[top_feats]+eps)/(signed_sum.sum()+eps)).abs()
		# top_acts_ratio_abs=((abs_sum[top_feats]+eps)/(abs_sum.sum()+eps)).abs()

		top_acts_ratio_abs=avg_activation_ratios[neuron,top_feats]
		top_acts_ratio_signed=avg_activation_ratios[neuron,top_feats]
	
		

		# Retrieve the top 3 most activating tokens for each feature
		top_tokens = []
		for feat_idx in top_feats:
			# Get activations for this feature
			feat_activations = preacts_BNH[:, neuron, feat_idx]
			# Get indices of top 3 activations
			top_indices = torch.topk(feat_activations, k=5).indices.tolist()
			# Map indices to tokens using tokenizer
			token_texts = [llm.tokenizer.decode([tokens[idx]]) for idx in top_indices]
			top_tokens.append(token_texts)

		# Append the data to table rows
		table_rows.append({
			'Neuron': neuron,
			'Top Features': top_feats,
			'Absolute Sum Ratio': [float(f'{x:.3g}') for x in top_acts_ratio_abs.tolist()],
			'Signed Sum Ratio': [float(f'{x:.3g}') for x in top_acts_ratio_signed.tolist()],
			'Top Tokens at neuron': top_tokens,
		})

	# Create DataFrame from the table rows
	df = pd.DataFrame(table_rows)

	# Sort the DataFrame by the ratio of the lead feature to the rest in descending order
	df['Lead Feature Ratio'] = df['Absolute Sum Ratio'].apply(lambda x: x[0] if len(x) > 0 else 0)
	df = df.sort_values(by='Lead Feature Ratio', ascending=False).reset_index(drop=True)

	# Optionally, select and rename relevant columns
	df = df[['Neuron', 'Top Features', 'Absolute Sum Ratio', 'Signed Sum Ratio', 'Top Tokens at neuron']]

	return df

#preacts_BNH = get_preacts_block(enc_acts,crosscoder,block)


def render_html_table(df: pd.DataFrame, input_text: str = None):
	pd.set_option('display.max_rows', 20)

	# Create HTML with CSS styling
	html = """
	<style>
	.scrollable-cell {
		max-height: 100px;
		overflow-y: auto;
		white-space: pre-wrap;
	}
	.input-text {
		margin: 20px 0;
		padding: 15px;
		background-color: #f5f5f5;
		border-radius: 5px;
		font-family: monospace;
		white-space: pre-wrap;
		line-height: 1.5;
	}
	.section-title {
		font-size: 1.2em;
		font-weight: bold;
		margin: 10px 0;
		color: #333;
	}
	</style>
	"""

	# Add input text section if provided
	if input_text:
		html += f"""
		<div class="section-title">Input Text:</div>
		<div class="input-text">{input_text}</div>
		<div class="section-title">Feature Analysis:</div>
		"""

	# Get first n rows and last row
	n = 10  # You can adjust this number
	n_back = 5
	df_display = pd.concat([df.head(n), df.tail(n_back)])

	# Convert DataFrame to HTML with scrollable cells
	html += df_display.to_html(classes='table', escape=False, render_links=True,
							formatters={
								'Top Features': lambda x: f'<div class="scrollable-cell">{x}</div>',
								'Absolute Sum Ratio': lambda x: f'<div class="scrollable-cell">{x}</div>', 
								'Signed Sum Ratio': lambda x: f'<div class="scrollable-cell">{x}</div>',
								'Top Tokens': lambda x: f'<div class="scrollable-cell">{x}</div>'
							})

	# Save and open the HTML file
	folder = '/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/data/sleeper_xcoders/analysis_data'
	file_path = f'{folder}/neuron_analysis.html'
	with open(file_path, 'w') as f:
		f.write(html)
	
	# Open the HTML file in the default web browser
	import webbrowser, os
	webbrowser.open('file://' + os.path.realpath(file_path))


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# load crosscoder decoder features

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




def get_dataloader_mean_SMPD(wandb_run_name:str):
	api = wandb.Api()
	artifact = api.artifact(f"dmitry2-uiuc/sleeper-model-diffing/dataloader-means_run-{wandb_run_name}:latest")
	artifact_dir = Path(artifact.download(root="../../.wandb_artifacts"))
	dataloader_mean_SMPD = torch.load(artifact_dir / "dataloader_means.pt", map_location=DEVICE)
	return dataloader_mean_SMPD



def main():
	torch.set_grad_enabled(False)
	
	checkpoint_dir = Path('../../.checkpoints/')
	wandb_run_name='1k68kpv5'#new l=1_000, bias=True

	
	
	crosscoder = load_crosscoder_from_wandb(
	"dmitry2-uiuc",
	"sleeper-model-diffing",
	wandb_run_name,
	"../../.wandb_artifacts",
	DEVICE)

	# dataloader_mean_SMPD = get_dataloader_mean_SMPD(wandb_run_name)

	from datasets import load_dataset

	dataset = load_dataset('mars-jason-25/tiny_stories_instruct_sleeper_data', split='train')
	dataset = dataset.filter(lambda x: x['is_training'] == True)



	
	llm = build_llm_lora(
		base_model_repo="roneneldan/TinyStories-Instruct-33M",
		lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
		cache_dir=None,
		device=DEVICE,
		dtype=None
	)
	tokenizer = llm.tokenizer


	

	input = dataset[0]['text']
	example_texts = [input]
	enc_acts,raw_acts = get_activations(input,llm,crosscoder)
	example_activations = [enc_acts]

	# p_BNH=get_preacts_nocontract(enc_acts,crosscoder.W_dec_HXD,crosscoder.b_dec_XD,llm,bias=False,block=0)
	# print(f'feat 782 acts: {enc_acts[:10,782]}')
	# print(f'feat. 782 max: {p_BNH[:,:,782]}')
	# exit()
	# p_NH_max=p_BNH.abs().max(dim=-1)[0]/p_BNH.abs().sum(dim=-1)
	# max_mean=p_NH_max.mean()
	# print(f'max_mean: {max_mean}')
	# exit()

	

	
	
	# feat_table=make_feat_table(llm,enc_acts,llm.tokenizer.encode(input),crosscoder)
	# render_html_table(feat_table, input_text=input)
	# exit()
	
	
	
	# Top occurring max feats - new XC
	# viz_dic={}
	# for feat_ind in [29,797,421,722,718]:
	# 	visualization=display_feature_activation_visualization(tokenizer, feature_index=feat_ind, example_texts=example_texts,example_activations=example_activations)
	# 	viz_dic[f"Feature: {str(feat_ind)}"]=visualization.data
	
	# serve_multiple_visualizations(viz_dic)
	
	# wandb_run_name_nopenalty='biv1u3ig'
	wandb_run_name_nopenalty='vl9klznb'
	crosscoder_nopenalty = load_crosscoder_from_wandb(
	"dmitry2-uiuc",
	"sleeper-model-diffing",
	wandb_run_name_nopenalty,
	"../../.wandb_artifacts",
	DEVICE)

	enc_acts_nopenalty,raw_acts_nopenalty = get_activations(input,llm,crosscoder_nopenalty)

	
	example_activations = [enc_acts_nopenalty]
	example_texts = [input]
	
	
	
	# Top occurring max feats - old XC
	viz_dic={}
	# for feat_ind in [493,782,132,1134,1147]:
	for feat_ind in [1230,913,1015,1227,1176]:
		visualization=display_feature_activation_visualization(tokenizer, feature_index=feat_ind, example_texts=example_texts,example_activations=example_activations)
		viz_dic[f"Feature: {str(feat_ind)}"]=visualization.data
	
	serve_multiple_visualizations(viz_dic)
	exit()

	enc_acts_nopenalty,raw_acts_nopenalty = get_activations(input,llm,crosscoder_nopenalty)
	example_activations_nopenalty = [enc_acts_nopenalty]

	visualization_nopenalty=display_feature_activation_visualization(tokenizer, feature_index=1480, example_texts=example_texts,example_activations=example_activations_nopenalty)
	#serve_multiple_visualizations({visualization_nopenalty.data: visualization_nopenalty.data})

	visualization=display_feature_activation_visualization(tokenizer, feature_index=1480, example_texts=example_texts,example_activations=example_activations)
	serve_multiple_visualizations({'new': visualization.data,'old': visualization_nopenalty.data})




	

if __name__ == '__main__':
	
	main()