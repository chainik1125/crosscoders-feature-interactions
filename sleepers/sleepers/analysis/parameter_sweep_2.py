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




import wandb
from typing import Dict, List, Tuple, Any, Optional


def extract_last_values_from_wandb_runs(
    entity: str,
    project: str,
    run_ids: List[str],
    metrics: List[str],
    filters: Optional[Dict[str, Any]] = None,
    make_int_mats:bool=False,
    dataset=None,
    llm=None
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

            if make_int_mats:
                feat_ints=get_interaction_metric(run_id,dataset,llm,sample_size=2)
                result_dict[key]['feat_ints']=feat_ints.mean()
            # Skip feat_ints entirely when make_int_mats=False
            
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


# extracted_data=extract_last_values_from_wandb_runs(
#     entity="dmitry2-uiuc",
#     project="sleeper-model-diffing",
#     run_ids=list(run_id_dic.values()),
#     metrics=["train/mean_unexplained_variance", "train/reconstruction_loss", "train/mean_max_ratio_mlp", "train/minus_max_mean", "train/minus_max_mean_loss"],
# )


def get_interaction_metric(crosscoder_ref,dataset,llm,sample_size=10):
    
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


def old_plot():
    explained_variance=[]
    mean_max_ratio=[]
    neuron_sharpness=[]
    reconstruction_loss=[]
    lambda_vals=[]
    feat_ints=[]
    penalty_loss=[]

    

    paper_run_id_dic={
    (10_000,1,0):'b5l291e5',
    (2_000,1,0):'bn1xtudv',
    (1_000,1,0):'ckubmeg1',
    (500,1,0):'x21ussr1',
    (200,1,0):'7avbfdww',
    (100,1,0):'vh2bylhi',
    (50,1,0):'bn2qo3w9',
    (20,1,0):'ni4z2dkr',
    (10,1,0):'hhm6y0s6',
    (0,1,0):'86u64trx',
        }
    

    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")

    #dataset = dataset.filter(lambda x: x['is_training'] == True)    

    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )

    extracted_data=extract_last_values_from_wandb_runs(
    entity="dmitry2-uiuc",
    project="sleeper-model-diffing",
    run_ids=list(paper_run_id_dic.values()),
    metrics=["train/mean_unexplained_variance", "train/reconstruction_loss", "train/mean_max_ratio_mlp", "train/minus_max_mean", "train/minus_max_mean_loss"],
    make_int_mats=True,
    dataset=dataset,
    llm=llm
    )



    #get interaction data
    
        



    for key,values in extracted_data.items():
        lambda_vals.append(key[0])
        explained_variance.append(1-values["train/mean_unexplained_variance"])
        mean_max_ratio.append(values["train/mean_max_ratio_mlp"])
        neuron_sharpness.append(values["train/minus_max_mean"])
        reconstruction_loss.append(values["train/reconstruction_loss"])
        penalty_loss.append(values["train/minus_max_mean_loss"])
        feat_ints.append(values["feat_ints"])



    
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


def extract_last_n_averaged_from_wandb_run(
    entity: str,
    project: str,
    run_id: str,
    metrics: list[str],
    n: int = 10
) -> dict[str, float]:
    """
    Extract the averaged last n values for specified metrics from a single wandb run.
    
    Args:
        entity: The wandb entity (username or team name)
        project: The wandb project name
        run_id: Run ID to extract data from
        metrics: List of metric names to extract
        n: Number of last values to average over
    
    Returns:
        A dictionary mapping metric names to their averaged last n values
    """
    api = wandb.Api()
    result = {}
    
    try:
        # Get the run
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get run history
        history = run.history()
        
        # Extract the last n values for each metric and average them
        for metric in metrics:
            if metric in history.columns:
                # Get the last n non-NaN values
                metric_values = history[metric].dropna()
                if len(metric_values) >= n:
                    last_n_values = metric_values.iloc[-n:]
                    result[metric] = last_n_values.mean()
                elif len(metric_values) > 0:
                    # If we have fewer than n values, average all available
                    result[metric] = metric_values.mean()
                else:
                    print(f"Warning: No valid data for metric '{metric}' in run {run_id}")
            else:
                print(f"Warning: Metric '{metric}' not found in run {run_id}")
        
        # Extract config parameters based on actual structure
        lambda_val = run.config.get('train', {}).get('lam_n')
        beta_val = run.config.get('train', {}).get('beta_n')
        hidden_dim = run.config.get('crosscoder', {}).get('hidden_dim')
        
        result['interaction_penalty'] = lambda_val
        result['hidden_dim'] = hidden_dim
        result['beta'] = beta_val
        
    except Exception as e:
        print(f"Error processing run {run_id}: {e}")
    
    return result


def extract_last_n_averaged_from_wandb_run_with_model(
    entity: str,
    project: str,
    run_id: str,
    metrics: list[str],
    n: int = 10
) -> dict[str, float]:
    """
    Extract the averaged last n values for specified metrics from a single wandb run,
    including model parameter information.
    
    Args:
        entity: The wandb entity (username or team name)
        project: The wandb project name
        run_id: Run ID to extract data from
        metrics: List of metric names to extract
        n: Number of last values to average over
    
    Returns:
        A dictionary mapping metric names to their averaged last n values,
        plus model parameter information
    """
    api = wandb.Api()
    result = {}
    
    try:
        # Get the run
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get run history
        history = run.history()
        
        # Extract the last n values for each metric and average them
        for metric in metrics:
            if metric in history.columns:
                # Get the last n non-NaN values
                metric_values = history[metric].dropna()
                if len(metric_values) >= n:
                    last_n_values = metric_values.iloc[-n:]
                    result[metric] = last_n_values.mean()
                elif len(metric_values) > 0:
                    # If we have fewer than n values, average all available
                    result[metric] = metric_values.mean()
                else:
                    print(f"Warning: No valid data for metric '{metric}' in run {run_id}")
            else:
                print(f"Warning: Metric '{metric}' not found in run {run_id}")
        
        # Extract config parameters based on actual structure
        lambda_val = run.config.get('train', {}).get('lam_n')
        beta_val = run.config.get('train', {}).get('beta_n')
        hidden_dim = run.config.get('crosscoder', {}).get('hidden_dim')
        
        # Extract model information - look for model size in different possible config locations
        model_params = None
        
        # Try to get model parameters from the config structure
        # Check if there's a model config section
        model_config = run.config.get('model', {})
        if model_config:
            # Look for common parameter indicators
            if 'n_params' in model_config:
                model_params = model_config['n_params']
            elif 'parameters' in model_config:
                model_params = model_config['parameters']
        
        # If not found in model config, try other locations
        if model_params is None:
            # Try direct config keys
            if 'n_params' in run.config:
                model_params = run.config['n_params']
            elif 'parameters' in run.config:
                model_params = run.config['parameters']
            elif 'model_size' in run.config:
                model_params = run.config['model_size']
        
        # If still not found, try to infer from model name or other config
        if model_params is None:
            # Look at the model name or path if available
            model_name = run.config.get('model_name', '') or run.config.get('base_model_repo', '')
            if '33M' in model_name:
                model_params = 33
            elif '124M' in model_name:
                model_params = 124
            elif '335M' in model_name:
                model_params = 335
        
        result['interaction_penalty'] = lambda_val
        result['hidden_dim'] = hidden_dim
        result['beta'] = beta_val
        result['model_params'] = model_params
        
    except Exception as e:
        print(f"Error processing run {run_id}: {e}")
    
    return result


def table_func_hidden(run_ids: list[str], n: int = 10):
    """
    Create a table comparing reconstruction loss and mean max ratio MLP across 
    different hidden dimensions and interaction penalties.
    
    Args:
        run_ids: List of wandb run IDs to process
        n: Number of last values to average over (default: 10)
    
    Returns:
        A formatted table with hidden dim vs interaction penalty
    """
    # Extract data from all runs
    entity = "dmitry2-uiuc"
    project = "sleeper-model-diffing"
    metrics = ["train/reconstruction_loss", "train/mean_max_ratio_mlp"]
    
    # Collect data from all runs
    data = []
    for run_id in run_ids:
        run_data = extract_last_n_averaged_from_wandb_run(entity, project, run_id, metrics, n)
        if run_data:  # Only add if we got valid data
            data.append(run_data)
    
    # Group data by hidden_dim and interaction_penalty
    table_data = {}
    
    for run_data in data:
        hidden_dim = run_data.get('hidden_dim')
        penalty = run_data.get('interaction_penalty')
        rec_loss = run_data.get('train/reconstruction_loss')
        mean_max_ratio = run_data.get('train/mean_max_ratio_mlp')
        
        if all(x is not None for x in [hidden_dim, penalty, rec_loss, mean_max_ratio]):
            if hidden_dim not in table_data:
                table_data[hidden_dim] = {}
            table_data[hidden_dim][penalty] = {
                'rec_loss': rec_loss,
                'mean_max_ratio': mean_max_ratio
            }
    
    # Create and display the table
    if not table_data:
        print("No valid data found for table creation")
        return None
    
    # Get all unique penalties and sort them
    all_penalties = set()
    for hidden_data in table_data.values():
        all_penalties.update(hidden_data.keys())
    sorted_penalties = sorted(all_penalties)
    
    # Get all unique hidden dims and sort them (descending)
    sorted_hidden_dims = sorted(table_data.keys(), reverse=True)
    
    # Print table header
    print("\nTable: Reconstruction Loss / Mean Max Ratio MLP")
    print("=" * 80)
    print(f"{'Hidden Dim':<12}", end="")
    for penalty in sorted_penalties:
        print(f"{'λ=' + str(penalty):<20}", end="")
    print()
    print("-" * 80)
    
    # Print table rows
    for hidden_dim in sorted_hidden_dims:
        print(f"{hidden_dim:<12}", end="")
        for penalty in sorted_penalties:
            if penalty in table_data[hidden_dim]:
                rec_loss = table_data[hidden_dim][penalty]['rec_loss']
                mean_max_ratio = table_data[hidden_dim][penalty]['mean_max_ratio']
                print(f"{int(round(rec_loss))}/{int(round(mean_max_ratio * 100))}%".ljust(20), end="")
            else:
                print("N/A".ljust(20), end="")
        print()
    
    return table_data


def table_func_model(run_ids: list[str], model_scaling_id_dic: dict = None, n: int = 10):
    """
    Create a table comparing reconstruction loss and mean max ratio MLP across 
    different model sizes and interaction penalties.
    
    Args:
        run_ids: List of wandb run IDs to process
        model_scaling_id_dic: Dictionary mapping (parameters, hidden_dim, penalty, seed) to run_ids
        n: Number of last values to average over (default: 10)
    
    Returns:
        A formatted table with model size vs interaction penalty
    """
    # Extract data from all runs
    entity = "dmitry2-uiuc"
    project = "sleeper-model-diffing"
    metrics = ["train/reconstruction_loss", "train/mean_max_ratio_mlp"]
    
    # Collect data from all runs
    data = []
    for run_id in run_ids:
        if not run_id:  # Skip empty run_ids
            continue
        run_data = extract_last_n_averaged_from_wandb_run_with_model(entity, project, run_id, metrics, n)
        if run_data:  # Only add if we got valid data
            # If we have the model_scaling_id_dic, use it to get model params
            if model_scaling_id_dic:
                for key, rid in model_scaling_id_dic.items():
                    if rid == run_id:
                        run_data['model_params'] = key[0]  # First element is model parameters
                        break
            data.append(run_data)
    
    # Group data by model parameters and interaction_penalty
    table_data = {}
    
    for run_data in data:
        penalty = run_data.get('interaction_penalty')
        rec_loss = run_data.get('train/reconstruction_loss')
        mean_max_ratio = run_data.get('train/mean_max_ratio_mlp')
        model_params = run_data.get('model_params')
        
        if all(x is not None for x in [penalty, rec_loss, mean_max_ratio, model_params]):
            if model_params not in table_data:
                table_data[model_params] = {}
            table_data[model_params][penalty] = {
                'rec_loss': rec_loss,
                'mean_max_ratio': mean_max_ratio
            }
    
    # Create and display the table
    if not table_data:
        print("No valid data found for table creation")
        return None
    
    # Get all unique penalties and sort them
    all_penalties = set()
    for model_data in table_data.values():
        all_penalties.update(model_data.keys())
    sorted_penalties = sorted(all_penalties)
    
    # Get all unique model params and sort them (descending)
    sorted_model_params = sorted(table_data.keys(), reverse=True)
    
    # Print table header
    print("\nTable: Reconstruction Loss / Mean Max Ratio MLP")
    print("=" * 80)
    print(f"{'Model':<20}", end="")
    for penalty in sorted_penalties:
        print(f"{'λ=' + str(penalty):<20}", end="")
    print()
    print("-" * 80)
    
    # Print table rows
    for model_params in sorted_model_params:
        # Format model name based on parameters
        if model_params == 335:
            model_name = "GPT2 Medium-335M"
        elif model_params == 124:
            model_name = "TinyStories-124M"
        elif model_params == 33:
            model_name = "TinyStories-33M"
        else:
            model_name = f"Model-{model_params}M"
        
        print(f"{model_name:<20}", end="")
        for penalty in sorted_penalties:
            if penalty in table_data[model_params]:
                rec_loss = table_data[model_params][penalty]['rec_loss']
                mean_max_ratio = table_data[model_params][penalty]['mean_max_ratio']
                print(f"{int(round(rec_loss))}/{int(round(mean_max_ratio * 100))}%".ljust(20), end="")
            else:
                print("N/A".ljust(20), end="")
        print()
    
    return table_data


def scaling_figure(xc_scaling_id_dict, model_scaling_id_dict, make_int_mats=False):
    """
    Create a scaling figure with two subplots showing how metrics change with penalty strength
    for different hidden dimensions (xc_scaling) and model parameters (model_scaling).
    
    Args:
        xc_scaling_id_dict: Dictionary with keys (hidden_dim, penalty, beta, seed) -> run_id
        model_scaling_id_dict: Dictionary with keys (model_params, hidden_dim, penalty, seed) -> run_id
        make_int_mats: Whether to calculate interaction matrices or load from files
    """
    
    # Load dataset and model for interaction calculations if needed
    if make_int_mats:
        dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
        llm = build_llm_lora(
            base_model_repo="roneneldan/TinyStories-Instruct-33M",
            lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
            cache_dir=None,
            device=DEVICE,
            dtype=None,
        )
    else:
        dataset = None
        llm = None
    
    # Extract data for crosscoder scaling (using individual run processing like table functions)
    entity = "dmitry2-uiuc"
    project = "sleeper-model-diffing"
    metrics = ["train/reconstruction_loss", "train/mean_max_ratio_mlp"]
    
    # Collect XC scaling data
    xc_data = []
    for key, run_id in xc_scaling_id_dict.items():
        if not run_id:  # Skip empty run_ids
            continue
        run_data = extract_last_n_averaged_from_wandb_run(entity, project, run_id, metrics, n=10)
        if run_data:  # Only add if we got valid data
            # Add the scaling dimensions to run_data
            hidden_dim, penalty, beta, seed = key
            run_data['hidden_dim'] = hidden_dim
            run_data['penalty'] = penalty
            xc_data.append(run_data)
    
    # Collect Model scaling data  
    model_data = []
    for key, run_id in model_scaling_id_dict.items():
        if not run_id:  # Skip empty run_ids
            continue
        run_data = extract_last_n_averaged_from_wandb_run_with_model(entity, project, run_id, metrics, n=10)
        if run_data:  # Only add if we got valid data
            # Add the scaling dimensions to run_data
            model_params, hidden_dim, penalty, seed = key
            run_data['model_params'] = model_params
            run_data['penalty'] = penalty
            model_data.append(run_data)
    
    # Create subplot figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Crosscoder Hidden Dimension Scaling", "Model Parameter Scaling"))
    
    # Define line styles and markers for different series
    line_styles = ['solid', 'dash', 'dot', 'dashdot']
    markers = ['circle', 'square', 'diamond', 'cross']
    xc_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    model_colors = ['darkgreen', 'darkred', 'darkorange', 'purple', 'brown', 'black']
    
    # Process crosscoder scaling data (subplot 1)
    xc_data_by_hidden_dim = {}
    for run_data in xc_data:
        hidden_dim = run_data.get('hidden_dim')
        penalty = run_data.get('penalty') 
        rec_loss = run_data.get('train/reconstruction_loss')
        mean_max_ratio = run_data.get('train/mean_max_ratio_mlp')
        
        if all(x is not None for x in [hidden_dim, penalty, rec_loss, mean_max_ratio]):
            if hidden_dim not in xc_data_by_hidden_dim:
                xc_data_by_hidden_dim[hidden_dim] = {'penalties': [], 'mean_max_ratios': [], 'reconstruction_losses': []}
            
            xc_data_by_hidden_dim[hidden_dim]['penalties'].append(penalty)
            xc_data_by_hidden_dim[hidden_dim]['mean_max_ratios'].append(mean_max_ratio)
            xc_data_by_hidden_dim[hidden_dim]['reconstruction_losses'].append(rec_loss)
    
    print(f"Final XC data by hidden dim: {list(xc_data_by_hidden_dim.keys())}")
    for hd, data in xc_data_by_hidden_dim.items():
        print(f"  Hidden dim {hd}: {len(data['penalties'])} points")
    
    # Sort data by penalty for each hidden dimension and add traces
    for i, (hidden_dim, data) in enumerate(sorted(xc_data_by_hidden_dim.items())):
        # Sort by penalty
        sorted_indices = np.argsort(data['penalties'])
        sorted_penalties = np.array(data['penalties'])[sorted_indices]
        sorted_ratios = np.array(data['mean_max_ratios'])[sorted_indices]
        sorted_losses = np.array(data['reconstruction_losses'])[sorted_indices]
        
        expansion_factor = hidden_dim / 1536
        series_name = f"Expansion {expansion_factor:.1f}x"
        
        fig.add_trace(go.Scatter(
            x=sorted_ratios,
            y=sorted_losses,
            mode="lines+markers",
            line=dict(dash=line_styles[i % len(line_styles)], color=xc_colors[i % len(xc_colors)], width=2),
            marker=dict(symbol=markers[i % len(markers)], size=10, color=xc_colors[i % len(xc_colors)]),
            name=series_name,
            text=[f"λ={p}, {series_name}" for p in sorted_penalties],
            hoverinfo="text",
        ), row=1, col=1)
    
    # Process model scaling data (subplot 2) 
    model_data_by_params = {}
    for run_data in model_data:
        model_params = run_data.get('model_params')
        penalty = run_data.get('penalty')
        rec_loss = run_data.get('train/reconstruction_loss')
        mean_max_ratio = run_data.get('train/mean_max_ratio_mlp')
        
        if all(x is not None for x in [model_params, penalty, rec_loss, mean_max_ratio]):
            if model_params not in model_data_by_params:
                model_data_by_params[model_params] = {'penalties': [], 'mean_max_ratios': [], 'reconstruction_losses': []}
            
            model_data_by_params[model_params]['penalties'].append(penalty)
            model_data_by_params[model_params]['mean_max_ratios'].append(mean_max_ratio)
            model_data_by_params[model_params]['reconstruction_losses'].append(rec_loss)
    
    print(f"Final Model data by params: {list(model_data_by_params.keys())}")
    for mp, data in model_data_by_params.items():
        print(f"  Model params {mp}: {len(data['penalties'])} points")
    
    # Add model scaling traces
    for i, (model_params, data) in enumerate(sorted(model_data_by_params.items())):
        # Sort by penalty
        sorted_indices = np.argsort(data['penalties'])
        sorted_penalties = np.array(data['penalties'])[sorted_indices]
        sorted_ratios = np.array(data['mean_max_ratios'])[sorted_indices]
        sorted_losses = np.array(data['reconstruction_losses'])[sorted_indices]
        
        # Create model-specific names
        if model_params == 33:
            model_name = "TinyStories-33M"
        elif model_params == 124:
            model_name = "TinyStories-124M"
        elif model_params == 335:
            model_name = "GPT2_335M (Openweb)"
        else:
            model_name = f"Model-{model_params}M"
        
        fig.add_trace(go.Scatter(
            x=sorted_ratios,
            y=sorted_losses,
            mode="lines+markers",
            line=dict(dash=line_styles[i % len(line_styles)], color=model_colors[i % len(model_colors)], width=2),
            marker=dict(symbol=markers[i % len(markers)], size=10, color=model_colors[i % len(model_colors)]),
            name=model_name,
            text=[f"λ={p}, {model_name}" for p in sorted_penalties],
            hoverinfo="text",
        ), row=1, col=2)
    
    # Apply the same formatting as old_plot row=1,col=1
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(180,180,180,0.9)",
        gridwidth=1,
        zeroline=False,
        ticks="outside",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(180,180,180,0.9)",
        gridwidth=1,
        zeroline=False,
        ticks="outside",
    )

    # Set axis titles
    fig.update_xaxes(title="Dominant feature share of L1 norm", row=1, col=1)
    fig.update_yaxes(title="Reconstruction Loss", row=1, col=1)
    fig.update_xaxes(title="Dominant feature share of L1 norm", row=1, col=2)
    fig.update_yaxes(title="Reconstruction Loss", row=1, col=2)

    # Apply layout styling matching old_plot
    fig.update_layout(
        plot_bgcolor="rgba(245,245,245,1)",
        paper_bgcolor="rgba(255,255,255,1)",
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=80, r=80, t=100, b=80),
        #legend_title_text="Series",
    )

    # Set font sizes
    fig.update_xaxes(tickfont=dict(size=24), title_font=dict(size=24))
    fig.update_yaxes(tickfont=dict(size=24), title_font=dict(size=24))
    fig.update_layout(legend=dict(font=dict(size=20)))
    
    # Save the figure
    filepath = "scaling_figure"
    pio.kaleido.scope.mathjax = False
    fig.write_image(filepath + '.pdf', engine='kaleido', width=2000, height=600)
    print(f'Scaling figure saved to: {filepath}.pdf')
    
    return fig

def sae_bench_figure(sae_data_dict):
    fig=make_subplots(rows=1,cols=1)

    int_penalties=[]
    AUCs=[]

    for run_key in sae_data_dict:
        int_penalty=run_key[1]
        auc=sae_data_dict[run_key]

        int_penalties.append(int_penalty)
        AUCs.append(auc)
    
    fig.add_trace(go.Bar(x=[f'λ={ip}' for ip in int_penalties],y=AUCs),row=1,col=1)

    fig.update_xaxes(title_text="Penalty strength")
    fig.update_yaxes(title_text="Area under curve (AUC)",range=[0,1])

    ignored_dir_path="/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/large_files/classifier/graphs"
    filepath=f'{ignored_dir_path}/sae_bench_sentiment.pdf'
    fig.write_image(filepath,engine='kaleido',width=2000,height=600)

    print(f'figure saved in: {filepath}')


if __name__ == "__main__":
    print('main character')
    

    xc_scaling_id_dic={
    (1536,0,1,0):'86u64trx',
    #(1536,10,1,0):'hhm6y0s6',
    #(1536,50,1,0):'bn2qo3w9',
    (1536,1_000,1,0):'ckubmeg1',
    #(1536,500,1,0):'x21ussr1',
    (1536,200,1,0):'7avbfdww',
    #(1536,100,1,0):'vh2bylhi',
    (1536,10_000,1,0):'b5l291e5',
    #
    (6144,0,1,0):'i8u5ddfm',
    (6144,200,1,0):'8w1kgrmt',
    (6144,1_000,1,0):'h4f3kf81',
    (6144,10_000,1,0):'sfr7vdbh',
    #
    (12_288,0,1,0):'76zgae5c',
    (12_288,200,1,0):'4rcf3zmx',
    (12_288,1_000,1,0):'33u2yrzn',
    (12_288,10_000,1,0):'eh8744o7',

    #(3072,0,1,0):''
}
    #(parameters,im_penalty,seed,0)
    model_scaling_id_dic={
        (33,1536,0,1):'86u64trx',
        (33,1536,200,0):'7avbfdww',
        (33,1536,1_000,0):'ckubmeg1',
        (33,1536,10_000,0):'b5l291e5',
        #
        # (124,1536,0,1):'86u64trx',
        # (124,1536,200,0):'7avbfdww',
        # (124,1536,1_000,0):'ckubmeg1',
        # (124,1536,10_000,0):'b5l291e5',
        #
        (124,3072,0,0):'3ijz8akp',
        (124,3072,200,0):'rbhzcxpk',
        (124,3072,1_000,0):'20lzmbbg',
        (124,3072,10_000,0):'jcdszwpi',
        #
        (335,1536,0,1):'lirv6aml',
        (335,1536,200,0):'gi3gzvqm',
        (335,1536,1_000,0):'fjcrr6kv',
        (335,1536,10_000,0):'9lufp9gm',

    }

    sae_bench_sentiment_results={
        (1536,0,1,0):0.60,
        (1536,200,1,0):0.58,
        (1536,1_000,1,0):0.58,
        (1536,10_000,1,0):0.59,
    }

    #run_ids = [run_id for run_id in xc_scaling_id_dic.values() if run_id]
    #table_data = table_func_hidden(run_ids, n=5)

    #model_run_ids = [run_id for run_id in model_scaling_id_dic.values() if run_id]
    #model_table_data = table_func_model(run_ids=model_run_ids,model_scaling_id_dic=model_scaling_id_dic,n=5)

    #old_plot()
    
    # Create scaling figure
    #scaling_figure(xc_scaling_id_dic, model_scaling_id_dic, make_int_mats=False)

    #SAE bench figure:

    sae_bench_figure(sae_bench_sentiment_results)
    