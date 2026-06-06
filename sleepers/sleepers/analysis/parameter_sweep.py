import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import torch
import pickle
import os,sys
import einops
from datasets import load_dataset
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from tqdm import tqdm
from sleepers.analysis.analysis_utils import (
    save_dict, 
    load_dict, 
    get_preacts_mlp, 
    get_activations, 
    feature_interactions_mlp, 
    get_preacts_nocontract_faster,
    get_preacts_nocontract,
    feature_interactions_sum,
    feature_interactions_alltokens
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



#old penalty
# run_id_dic={
#     (100,1,0):"ukqpfwjb",
#     (50,1,0):"24u342lh",
#     (20,1,0):"ezn57gko",
#     (10,1,0):"i8kwi4rl",
#     (0,1,0):"q3qpomq9",
#     (1,1,0):"02q4t69k",
#     (2.7,1,0):"i0mypno4",
#     (7.4,1,0):"fiwf9l79",
#     (54,1,0):"b1cu4gm0",

# }

# run_id_dic={
#     (10_000,1,0): 'ivnwngpj',
#     (1000,1,0):'3chdgves',
#     (500,1,0):'ve9w4sf0',
#     (200,1,0):'m7knhqp9',
#     (100,1,0): '3q7ooszt',
#     (50,1,0):'12jwmppc',
#     (10,1,0):'osnrhkrp',
#     (1,1,0):'k6bx3gcy',
#     (0,1,0):'biv1u3ig'
# }

#Very high compression achieved here, not sure how that's even possible - since the 1k68 XC is in there its probably with bias?
# run_id_dic={
#     (10_000,1,0):'hffcbpi2',
#     (1_000,1,0):'1k68kpv5',
#     (500,1,0):'o31pune1',
#     (200,1,0):'qf87htr2',
#     (10,1,0):'g2zxiqdf',
#     (0,1,0):'vl9klznb',
# }


#PAPER VALUES
# run_id_dic={
#     (10_000,1,0):'b5l291e5',
#     (2_000,1,0):'bn1xtudv',
#     (1_000,1,0):'ckubmeg1',
#     (500,1,0):'x21ussr1',
#     (200,1,0):'7avbfdww',
#     (100,1,0):'vh2bylhi',
#     (50,1,0):'bn2qo3w9',
#     (20,1,0):'ni4z2dkr',
#     (10,1,0):'hhm6y0s6',
#     (0,1,0):'86u64trx',
# }


#scaling 

#(hidden_dim,penalty,seed,0)
xc_scaling_id_dic={
    (1536,0,1,0):'86u64trx',
    
}



import wandb
from typing import Dict, List, Tuple, Any, Optional


def extract_last_values_from_wandb_runs(
    entity: str,
    project: str,
    run_ids: List[str],
    metrics: List[str],
    filters: Optional[Dict[str, Any]] = None
) -> Dict[Tuple, Dict[str, float]]:
    """
    Extract the last recorded values for specified metrics from wandb runs.
    
    Args:
        entity: The wandb entity (username or team name)
        project: The wandb project name
        run_ids: List of run IDs to extract data from
        metrics: List of metric names to extract
        filters: Optional dictionary of config parameters to filter runs by
    
    Returns:
        A dictionary where:
        - keys are tuples of (lambda, beta, seed) from the run config
        - values are dictionaries mapping metric names to their last recorded values
    """
    api = wandb.Api()
    result_dict = {}
    
    for run_id in run_ids:
        try:
            # Get the run
            run = api.run(f"{entity}/{project}/{run_id}")
            
            # Skip if filters don't match
            if filters:
                skip_run = False
                for key, value in filters.items():
                    if key not in run.config or run.config[key] != value:
                        skip_run = True
                        break
                if skip_run:
                    continue
            
            # Extract lambda, beta, seed from config
            lambda_val = run.config.get('train', None)['lam_n']
            beta_val = run.config.get('train', None)['beta_n']
            seed_val = run.config.get('seed', None)

            
            print(lambda_val, beta_val, seed_val)
            
            
            # Skip if any required config is missing
            if None in (lambda_val, beta_val, seed_val):
                print(f"Warning: Run {run_id} missing required config parameters")
                continue
            
            # Create the key tuple
            key = (lambda_val, beta_val, seed_val)
            
            # Extract the last values for each metric
            values = {}
            history = run.history()
            
            for metric in metrics:
                if metric in history.columns:
                    # Get the last non-NaN value
                    metric_values = history[metric].dropna()
                    if not metric_values.empty:
                        values[metric] = metric_values.iloc[-1]
                    else:
                        print(f"Warning: No valid data for metric '{metric}' in run {run_id}")
                else:
                    print(f"Warning: Metric '{metric}' not found in run {run_id}")
            
            # Add to result dictionary
            result_dict[key] = values

            feat_ints=pickle.load(open(f"feat_ints_samples_2/{run_id}.pkl", "rb"))
            result_dict[key]['feat_ints']=feat_ints.mean()
            
        except Exception as e:
            print(f"Error processing run {run_id}: {e}")
        

    
    return result_dict

#make the feat ints into a func
# sample_size=2
# base_dir=f"feat_ints_samples_{sample_size}"
# os.makedirs(base_dir,exist_ok=True)
# #test_fts=get_interaction_metric(run_id_dic[(10_000,1,0)])
# for run_id in tqdm(run_id_dic.values()):
#     test_fts=get_interaction_metric(run_id,sample_size)
#     save_path=f"{base_dir}/{run_id}.pkl"
#     pickle.dump(test_fts, open(save_path, "wb"))
#     print(f'saved to {save_path}')
# sys.exit()


extracted_data=extract_last_values_from_wandb_runs(
    entity="dmitry2-uiuc",
    project="sleeper-model-diffing",
    run_ids=list(run_id_dic.values()),
    metrics=["train/mean_unexplained_variance", "train/reconstruction_loss", "train/mean_max_ratio_mlp", "train/minus_max_mean", "train/minus_max_mean_loss"],
)


explained_variance=[]
mean_max_ratio=[]
neuron_sharpness=[]
reconstruction_loss=[]
lambda_vals=[]
feat_ints=[]
penalty_loss=[]

for key,values in extracted_data.items():
    lambda_vals.append(key[0])
    explained_variance.append(1-values["train/mean_unexplained_variance"])
    mean_max_ratio.append(values["train/mean_max_ratio_mlp"])
    neuron_sharpness.append(values["train/minus_max_mean"])
    reconstruction_loss.append(values["train/reconstruction_loss"])
    penalty_loss.append(values["train/minus_max_mean_loss"])
    feat_ints.append(values["feat_ints"])


dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")

#dataset = dataset.filter(lambda x: x['is_training'] == True)    

llm = build_llm_lora(
    base_model_repo="roneneldan/TinyStories-Instruct-33M",
    lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
    cache_dir=None,
    device=DEVICE,
    dtype=None,
)
def get_interaction_metric(crosscoder_ref,sample_size=10):
    
    base_dir=os.makedirs(f"feat_ints_samples_{sample_size}",exist_ok=True)
    crosscoder=load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", crosscoder_ref, "../../.wandb_artifacts", DEVICE
    )

    with torch.no_grad():
        feat_ints=torch.zeros((crosscoder.W_dec_HXD.shape[0],crosscoder.W_dec_HXD.shape[0]))
        for layer in range(4):
            feat_ints+=feature_interactions_sum(layer,sample_size,dataset,llm,crosscoder)
        feat_ints=feat_ints.cpu().numpy()
        

        #np.save(f"feat_ints_{crosscoder_ref}.npy",feat_ints)

    return feat_ints


    
    


fig = make_subplots(rows=1, cols=2)

# Create a color scale based on lambda values
lambda_array = np.array(lambda_vals)

# Create a list of colors for the points
colors = []
for lam in lambda_vals:
    if np.isclose(lam, 0):
        # Red for lambda=0
        colors.append('rgb(255,0,0)')
    else:
        # Interpolate from pale blue to dark blue based on log-scaled lambda value
        # Find the min and max lambda values (excluding 0)
        non_zero_lambdas = [l for l in lambda_vals if not np.isclose(l, 0)]
        min_lambda = min(non_zero_lambdas) if non_zero_lambdas else 1
        max_lambda = max(non_zero_lambdas) if non_zero_lambdas else 100
        
        # Log-scale the lambda value and normalize between 0 and 1
        log_lam = np.log10(lam) if lam > 0 else 0
        log_min = np.log10(min_lambda) if min_lambda > 0 else 0
        log_max = np.log10(max_lambda) if max_lambda > 0 else 2
        
        normalized = (log_lam - log_min) / (log_max - log_min) if log_max > log_min else 0.5
        
        # Interpolate between pale blue (173, 216, 230) and dark blue (0, 0, 139)
        r = int(173 - normalized * 173)
        g = int(216 - normalized * 216)
        b = int(230 - normalized * (230 - 139))
        
        colors.append(f'rgb({r},{g},{b})')

# Create the scatter plot
fig.add_trace(go.Scatter(
    x=np.array(mean_max_ratio),
    y=np.array(reconstruction_loss),
    mode="markers", 
    marker=dict(
        size=15,
        color=colors,
    ),
    text=[f"Lambda: {lam}" for lam in lambda_vals],  # Show actual lambda values instead of integers
    hoverinfo="text",
    showlegend=False
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=np.array(feat_ints),
    y=np.array(reconstruction_loss),
    mode="markers", 
    marker=dict(
        size=15,
        color=colors,
    ),
    text=[f"Lambda: {lam}" for lam in lambda_vals],  # Show actual lambda values instead of integers
    hoverinfo="text",
    showlegend=False
), row=1, col=2)

# fig.add_trace(go.Scatter(
#     x=np.array(penalty_loss),
#     y=np.array(reconstruction_loss),
#     mode="markers", 
#     marker=dict(
#         size=15,
#         color=colors,
#     ),
#     text=[f"Lambda: {lam}" for lam in lambda_vals],  # Show actual lambda values instead of integers
#     hoverinfo="text",
#     showlegend=False
# ), row=1, col=2)

# Add individual points to the legend for each lambda value


unique_lambdas = sorted(list(set(lambda_vals)))  # Use actual lambda values, not integers
for lam in unique_lambdas:
    idx = lambda_vals.index(lam)
    color = colors[idx]
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            size=15,
            color=color,
        ),
        name=f"λ = {lam:.0f}",  # Show actual lambda value
        showlegend=True,
        legendgrouptitle_font=dict(size=24),  # Increase legend font size
        legendgroup=f"lambda_{lam}",
    ))

# Find the index of lambda=0 point
# lambda_zero_idx = np.where(np.isclose(lambda_array, 0))[0]
# if len(lambda_zero_idx) > 0:
#     lambda_zero_idx = lambda_zero_idx[0]
#     # Add annotation with arrow pointing to the lambda=0 point
#     fig.add_annotation(
#         x=mean_max_ratio[lambda_zero_idx],
#         y=reconstruction_loss[lambda_zero_idx],
#         text="Conventional crosscoders",
#         showarrow=True,
#         arrowhead=2,
#         arrowsize=1.5,
#         arrowwidth=2,
#         arrowcolor="black",
#         font=dict(size=24, color="black"),
#         ax=0,
#         ay=-50
#     )

    # ------------------------------------------------------------------
# 2) global axis styling – one call affects every subplot
# ------------------------------------------------------------------
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(180,180,180,0.9)",   # same dark-grey grid
    gridwidth=1,
    zeroline=False,                      # remove bold x=0 line (optional)
    ticks="outside",
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(180,180,180,0.9)",
    gridwidth=1,
    zeroline=False,
    ticks="outside",
)

# ------------------------------------------------------------------
# 3) titles for each specific panel (keep these if you like)
# ------------------------------------------------------------------
fig.update_xaxes(title="Dominant feature share of L1 norm", row=1, col=1)
fig.update_yaxes(title="Reconstruction Loss",              row=1, col=1)

fig.update_xaxes(title="Average pairwise interaction",                     row=1, col=2)
fig.update_yaxes(title="Reconstruction Loss",              row=1, col=2)

# fig.update_xaxes(title="Average pairwise interaction",     row=1, col=3)
# fig.update_yaxes(title="Reconstruction Loss",              row=1, col=3)

# ------------------------------------------------------------------
# 4) common background / layout – already applies to every subplot
# ------------------------------------------------------------------
fig.update_layout(
    plot_bgcolor="rgba(245,245,245,1)",   # light-grey plot area
    paper_bgcolor="rgba(255,255,255,1)",  # white surrounding “paper”
    font=dict(family="Arial, sans-serif"),
    margin=dict(l=80, r=80, t=100, b=80),
    legend_title_text="Penalty strength",
)

# ------------------------------------------------------------------
# 5) font sizes
# ------------------------------------------------------------------
fig.update_xaxes(tickfont=dict(size=24), title_font=dict(size=24))
fig.update_yaxes(tickfont=dict(size=24), title_font=dict(size=24))
fig.update_layout(legend=dict(font=dict(size=20)))

# fig.update_xaxes(title="Dominant feature share of L1 norm", row=1, col=1)
# fig.update_yaxes(title="Reconstruction Loss", row=1, col=1)

# fig.update_xaxes(title="Norm penalty", row=1, col=2)
# fig.update_yaxes(title="Reconstruction Loss", row=1, col=2)

# fig.update_xaxes(title="Average pairwise interaction", row=1, col=3)
# fig.update_yaxes(title="Reconstruction Loss", row=1, col=3)

# fig.update_layout(
#     legend_title_text="Penalty strength",
#     plot_bgcolor='rgba(245,245,245,1)',  # Light gray background
#     paper_bgcolor='rgba(255,255,255,1)',  # White paper background
#     xaxis=dict(
#         showgrid=True,
#         gridcolor='rgba(180,180,180,0.9)',  # Darker gray gridlines
#         gridwidth=1
#     ),
#     yaxis=dict(
#         showgrid=True,
#         gridcolor='rgba(180,180,180,0.9)',  # Darker gray gridlines
#         gridwidth=1
#     ),
#     font=dict(family="Arial, sans-serif"),  # Clean font for poster readability
#     margin=dict(l=80, r=80, t=100, b=80)  # Generous margins
# )

# fig.update_xaxes(tickfont=dict(size=24),title_font=dict(size=24))
# fig.update_yaxes(tickfont=dict(size=24),title_font=dict(size=24))

filepath="sweep"
#"/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/graphs/loss_vs_neuron_sharpness.pdf"

fig2=make_subplots(rows=1,cols=1)
#fig2.write_image(filepath+'burner.pdf',width=2000,height=600)
pio.kaleido.scope.mathjax = False
fig.write_image(filepath+'.pdf',engine='kaleido',width=2000,height=600)
print(f'feat ints:\n {feat_ints}')
#fig2.show()

#fig.write_image(filepath,width=1000,height=1000)
#print(f"Graph saved to:\n {filepath}")

#fig.write_image(filepath.split('.pdf')[0]+'_2.pdf',width=1000,height=1000)
#print(f"Graph saved to:\n {filepath}")

#fig.show()
