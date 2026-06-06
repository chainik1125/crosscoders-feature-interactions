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
import os,sys
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
    propagate_preacts
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

# -----------------------------------------------------------------------------
# ---------------------- Helper visualisation functions -----------------------
# -----------------------------------------------------------------------------

def visualize_feature_stats_from_tensor(feature_stats_dict: dict, top_n: int = 50):
    """Same as before – unchanged."""
    neurons, features = feature_stats_dict["max_count"].shape

    total_max_counts = feature_stats_dict["max_count"].sum(dim=0)
    top_indices = torch.argsort(total_max_counts, descending=True)[:top_n]
    feature_indices = [f"F{idx}" for idx in top_indices.cpu().numpy()]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {"secondary_y": False}], [            
            {"secondary_y": False}, {"secondary_y": False}
        ]],
        subplot_titles=(
            "Feature Frequency",                          
            "Positive vs Negative Activations",
            "Max Features per Neuron",
            "Max Features per Datapoint",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=feature_indices,
            y=total_max_counts[top_indices].cpu().numpy(),
            name="Max Count",
            marker_color="royalblue",
        ),
        row=1,
        col=1,
    )

    pos_counts = (
        feature_stats_dict["max_plus_count"].sum(dim=0)[top_indices].cpu().numpy()
    )
    neg_counts = (
        feature_stats_dict["max_minus_count"].sum(dim=0)[top_indices].cpu().numpy()
    )

    fig.add_trace(
        go.Bar(x=feature_indices, y=pos_counts, name="Positive", marker_color="green"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=feature_indices, y=neg_counts, name="Negative", marker_color="red"),
        row=1,
        col=2,
    )

    max_features_per_neuron = (feature_stats_dict["max_count"] > 0).sum(dim=1)
    fig.add_trace(
        go.Histogram(
            x=max_features_per_neuron.cpu().numpy(),
            name="Features per Neuron",
            marker_color="purple",
            nbinsx=30,
        ),
        row=2,
        col=1,
    )

    features_per_datapoint = np.array(feature_stats_dict["max_features_per_datapoint"])

    fig.add_trace(
        go.Histogram(
            x=features_per_datapoint,
            name="Max Features per Datapoint",
            marker_color="orange",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        title="Feature Statistics Analysis from Tensor Data",
        legend=dict(x=1.0, y=1.0),
    )

    fig.update_xaxes(title_text="Feature", row=1, col=1)
    fig.update_xaxes(title_text="Feature", row=1, col=2)
    fig.update_xaxes(title_text="Number of Unique Features", row=2, col=1)
    fig.update_xaxes(title_text="Max Features per Datapoint", row=2, col=2)

    fig.update_yaxes(title_text="Occurrence Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Number of Neurons", row=2, col=1)
    fig.update_yaxes(title_text="Feature frequency", row=2, col=2)

    return fig


# -----------------------------------------------------------------------------
# ------------- collect one HTML <section> per feature ------------------------
# -----------------------------------------------------------------------------

def visualize_text_feature(feature_index: int, dataset, llm, crosscoder, get_activations_fns, number_of_examples=3):
    """Return a <section>…</section> string for a single feature; supports multiple highlight functions side by side."""
    # Ensure get_activations_fns is a list of functions
    if not isinstance(get_activations_fns, (list, tuple)):
        get_activations_fns = [get_activations_fns]
    show_labels = len(get_activations_fns) > 1
    # Derive labels for each function
    fn_names = [fn.__name__ for fn in get_activations_fns]

    # Pull example texts
    texts = [dataset[i]["text"] for i in range(number_of_examples)]
    # Build rows of highlighted snippets for each example
    example_rows = []
    for txt in texts:
        snippet_parts = []
        for fn, name in zip(get_activations_fns, fn_names):
            # Compute activations and render HTML
            act = fn(txt, llm, crosscoder)
            html_obj = display_feature_activation_visualization(
                llm.tokenizer,
                example_texts=[txt],
                example_activations=[act],
                feature_index=feature_index,
            )
            # Wrap each snippet (with label if multiple functions)
            if show_labels:
                snippet_html = (
                    f"<div style='flex:1; text-align:center; padding:5px;'>"
                    f"<h4>{name}</h4>{html_obj.data}</div>"
                )
            else:
                snippet_html = f"<div style='flex:1; padding:5px;'>{html_obj.data}</div>"
            snippet_parts.append(snippet_html)
        # Combine snippets side by side
        row_html = f"<div style='display:flex; gap:10px; margin-bottom:20px;'>{''.join(snippet_parts)}</div>"
        example_rows.append(row_html)

    # Wrap in <section>
    section_html = f"""
    <section id="feature_{feature_index}" style="display:none">
        <h2 style="margin-top:0;">Feature {feature_index}</h2>
        {''.join(example_rows)}
    </section>
    """
    return section_html


# -----------------------------------------------------------------------------
# ----------- glue everything into one self‑contained HTML file ---------------
# -----------------------------------------------------------------------------

def build_dashboard(tensor_data, dataset, llm, crosscoder, get_acts_fn, top_k=10,number_of_examples=3):
    # 1) Overview stats figure
    stats_fig = visualize_feature_stats_from_tensor(tensor_data)
    stats_html = stats_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # 2) Top‑k features by frequency
    max_count = tensor_data["max_count"].sum(dim=0)
    top_feats = torch.argsort(max_count, descending=True)[:top_k]

    feature_sections = []
    nav_links = []
    
    for idx in top_feats:
        idx_int = idx.item()
        feature_sections.append(
            visualize_text_feature(idx_int, dataset, llm, crosscoder, get_acts_fn,number_of_examples)
        )
        nav_links.append(
            f'<li><a href="#" onclick="show(\'feature_{idx_int}\')">Feature {idx_int}</a></li>'
        )

    # 3) Single HTML document
    full_html = dedent(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Feature‑activation dashboard</title>
            <style>
                body      {{ margin:0; font-family:system-ui,sans-serif; }}
                nav       {{ width:220px; position:fixed; top:0; left:0; bottom:0;
                             overflow:auto; background:#f8f8f8; padding:20px 10px; }}
                nav ul    {{ list-style:none; padding:0; }}
                nav li a  {{ text-decoration:none; color:#0366d6; display:block; padding:4px 0; }}
                section   {{ margin-left:240px; padding:20px; }}
            </style>
            <script>
                function hideAll() {{
                    document.querySelectorAll('section').forEach(s => s.style.display='none');
                }}
                function show(id) {{
                    hideAll();
                    document.getElementById(id).style.display='block';
                }}
                window.onload = () => {{
                    hideAll();
                    document.getElementById('overview').style.display='block';
                }};
            </script>
        </head>
        <body>
            <nav>
                <ul>
                    <li><a href="#" onclick="show('overview')"><strong>Overview</strong></a></li>
                    {''.join(nav_links)}
                </ul>
            </nav>

            <section id="overview">
                <h2 style="margin-top:0;">Overall feature statistics</h2>
                {stats_html}
            </section>

            {''.join(feature_sections)}
        </body>
        </html>
        """
    )

    out_path = pathlib.Path(tempfile.gettempdir()) / "feature_dashboard.html"
    out_path = "./feature_dashboard.html" # Save in current directory
    # Use pathlib for writing
    out_path_obj = pathlib.Path(out_path)
    out_path_obj.write_text(full_html, encoding="utf-8") 
    # webbrowser.open(out_path.as_uri()) # <<< Commented out this line
    print(f"Dashboard written to {out_path_obj.resolve()}") # Print absolute path



def visualize_text_int(rank_idx,story_indices,row_idx,col_idx,dataset, llm, crosscoder, get_activations_fns, number_of_examples=2):
    """Return a <section>…</section> string for a single feature; supports multiple highlight functions side by side."""
    

    # Pull example texts
    ranked_story_indices=story_indices[rank_idx][:number_of_examples]
    
    
    
    texts = [dataset[int(ranked_story_indices[i])]["text"] for i in range(number_of_examples)]
    # Build rows of highlighted snippets for each example
    example_rows = []
    for txt in texts:
        snippet_parts = []
        # Compute activations and render HTML
        act = feature_interactions_mlp(txt,llm,crosscoder,block=1)[:,row_idx,col_idx]
        
        acts_format=torch.zeros((128,1536))
        acts_format[:,row_idx]=act
        
        html_obj = display_feature_activation_visualization(
            llm.tokenizer,
            example_texts=[txt],
            example_activations=[acts_format],
            feature_index=row_idx,
            int_metric_label=True
        )
        # Wrap each snippet (with label if multiple functions)
        # if show_labels:
        #     snippet_html = (
        #         f"<div style='flex:1; text-align:center; padding:5px;'>"
        #         f"<h4>{name}</h4>{html_obj.data}</div>"
        #     )
        # else:
        snippet_html = f"<div style='flex:1; padding:5px;'>{html_obj.data}</div>"
        snippet_parts.append(snippet_html)

        #Now you want to do the same thing for the features
        feat_acts=get_activations(txt,llm,crosscoder)[0]
        #feat_mlp_acts=get_preacts_mlp(txt,llm,crosscoder,block=1)[0]#0 is block
        
        
        html_obj_feat=display_feature_activation_visualization(
            llm.tokenizer,
            example_texts=[txt],
            example_activations=[feat_acts],
            feature_index=row_idx,
            int_metric_label=False
        )
        snippet_html_feat = f"<div style='flex:1; padding:5px;'>{html_obj_feat.data}</div>"
        snippet_parts.append(snippet_html_feat)

        feat_acts=get_activations(txt,llm,crosscoder)[0]
        html_obj_feat=display_feature_activation_visualization(
            llm.tokenizer,
            example_texts=[txt],
            example_activations=[feat_acts],
            feature_index=col_idx,
            int_metric_label=False
        )
        snippet_html_feat = f"<div style='flex:1; padding:5px;'>{html_obj_feat.data}</div>"
        snippet_parts.append(snippet_html_feat)

        

        # Combine snippets side by side
        row_html = f"<div style='display:flex; gap:10px; margin-bottom:20px;'>{''.join(snippet_parts)}</div>"
        example_rows.append(row_html)

    # Wrap in <section>
    section_html = f"""
    <section id="feature_int_{rank_idx}" style="display:none">
        <h2 style="margin-top:0;">Feature {row_idx} & {col_idx}</h2>
        {''.join(example_rows)}
    </section>
    """
    return section_html

# def visualize_text_int(
#     rank_idx: int,
#     row_idx: int,
#     col_idx: int,
#     dataset,
#     llm,
#     crosscoder,
#     number_of_examples: int = 2,
# ):
#     """
#     Show the top-N examples of the (row_idx, col_idx) interaction.
#     On each example we render three side‑by‑side highlights:
#       1) the interaction strength at (row_idx, col_idx)
#       2) the row feature activation
#       3) the col feature activation
#     """
#     texts = [dataset[i]["text"] for i in range(number_of_examples)]
#     rows_html = []

#     for txt in texts:
#         # --- 1) get per‑token interaction scores at (row_idx, col_idx) ---
#         ints_all = feature_interactions_mlp(
#             txt, llm, crosscoder, block=1        # <-- named parameter so block=1 actually takes effect
#         )                                        # shape [seq_len, num_feats, num_feats]
#         ints_tok = ints_all[:, row_idx, col_idx] \
#                       .detach().cpu().numpy()     # shape [seq_len]

#         # --- 2) get the two individual feature activations ---
#         acts = get_activations(txt, llm, crosscoder)[0]  # shape [seq_len, num_feats]
#         row_tok = acts[:, row_idx].detach().cpu().numpy()
#         col_tok = acts[:, col_idx].detach().cpu().numpy()

#         # --- 3) encode to tokens (must match len(ints_tok)) ---
#         tokens = llm.tokenizer.encode(txt)[: len(ints_tok)]

#         # --- 4) render three side‑by‑side snippets via the low‑level highlighter ---
#         html_int = display_text_with_highlighting(
#             tokens, llm.tokenizer, ints_tok, transparent_test=lambda v: v == 0
#         )
#         html_row = display_text_with_highlighting(
#             tokens, llm.tokenizer, row_tok, transparent_test=lambda v: v == 0
#         )
#         html_col = display_text_with_highlighting(
#             tokens, llm.tokenizer, col_tok, transparent_test=lambda v: v == 0
#         )

#         rows_html.append(
#             f"<div style='display:flex; gap:10px; margin-bottom:20px;'>"
#             f"{html_int.data}{html_row.data}{html_col.data}"
#             f"</div>"
#         )

#     section_html = f"""
#     <section id="feature_int_{rank_idx}" style="display:none">
#       <h2>Interaction {row_idx} → {col_idx}</h2>
#       {''.join(rows_html)}
#     </section>
#     """
#     return section_html

def feat_int_dashboard(tensor_data,feat_int_data, dataset, llm, crosscoder, get_acts_fn, top_k=3,number_of_examples=2):
    assert number_of_examples <= feat_int_data.shape[0], "Precalculated feat_int_data has less datapoints than requested number of examples."
    # 1) Overview stats figure
    stats_fig = visualize_feature_stats_from_tensor(tensor_data)
    stats_html = stats_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # 2) I need to find
    def get_top_k_interactions(feat_int_data, top_k):
        #first find the top interactions when summed across all datapoints
        top_k_interactions = np.argsort(feat_int_data.sum(axis=0).flatten())[::-1][:top_k]
        rows, cols = np.unravel_index(top_k_interactions, feat_int_data[0,:,:].shape)
        #now, for each interaction, I want to find the indices of stories with largest values
        story_indices = np.zeros((len(rows), number_of_examples), dtype=int)
        for i, (r, c) in enumerate(zip(rows, cols)):
            story_indices[i] = np.argsort(feat_int_data[:, r, c])[::-1][:number_of_examples]
        return rows, cols, story_indices
    
    rows, cols, story_indices = get_top_k_interactions(feat_int_data, top_k)
    
    feature_sections = []
    nav_links = []

    #hack to add the int. visualization

            
    
    for max_interaction_idx, top_row_idx in enumerate(rows):
        top_row_idx,top_col_idx = rows[max_interaction_idx], cols[max_interaction_idx]
        feature_sections.append(
            visualize_text_int(max_interaction_idx,story_indices,top_row_idx,top_col_idx,dataset, llm, crosscoder, get_acts_fn,number_of_examples)
        )
        # raise Exception("Stop here")
        nav_links.append(
            f'<li><a href="#" onclick="show(\'feature_int_{max_interaction_idx}\')">Int. {top_row_idx} & {top_col_idx}</a></li>'
        )
        # feature_sections.append(
        #     visualize_text_feature(top_row_idx, dataset, llm, crosscoder, get_acts_enc,number_of_examples)
        # )
        # nav_links.append(
        #     f'<li><a href="#" onclick="show(\'feature_{top_row_idx}\')">Feature {top_row_idx}</a></li>'
        # )

        # feature_sections.append(
        #     visualize_text_feature(top_col_idx, dataset, llm, crosscoder, get_acts_enc,number_of_examples)
        # )
        # nav_links.append(
        #     f'<li><a href="#" onclick="show(\'feature_{top_col_idx}\')">Feature {top_col_idx}</a></li>'
        # )


    # 3) Single HTML document
    full_html = dedent(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Feature‑activation dashboard</title>
            <style>
                body      {{ margin:0; font-family:system-ui,sans-serif; }}
                nav       {{ width:220px; position:fixed; top:0; left:0; bottom:0;
                             overflow:auto; background:#f8f8f8; padding:20px 10px; }}
                nav ul    {{ list-style:none; padding:0; }}
                nav li a  {{ text-decoration:none; color:#0366d6; display:block; padding:4px 0; }}
                section   {{ margin-left:240px; padding:20px; }}
            </style>
            <script>
                function hideAll() {{
                    document.querySelectorAll('section').forEach(s => s.style.display='none');
                }}
                function show(id) {{
                    hideAll();
                    document.getElementById(id).style.display='block';
                }}
                window.onload = () => {{
                    hideAll();
                    document.getElementById('overview').style.display='block';
                }};
            </script>
        </head>
        <body>
            <nav>
                <ul>
                    <li><a href="#" onclick="show('overview')"><strong>Overview</strong></a></li>
                    {''.join(nav_links)}
                </ul>
            </nav>

            <section id="overview">
                <h2 style="margin-top:0;">Overall feature statistics</h2>
                {stats_html}
            </section>

            {''.join(feature_sections)}
        </body>
        </html>
        """
    )

    out_path = pathlib.Path(tempfile.gettempdir()) / "feature_dashboard.html"
    out_path = "./feature_dashboard.html" # Save in current directory
    # Use pathlib for writing
    out_path_obj = pathlib.Path(out_path)
    out_path_obj.write_text(full_html, encoding="utf-8") 
    # webbrowser.open(out_path.as_uri()) # <<< Commented out this line
    print(f"Dashboard written to {out_path_obj.resolve()}") # Print absolute path

# -----------------------------------------------------------------------------
# ------------------------------ main entry -----------------------------------
# -----------------------------------------------------------------------------

def get_ft_int_SH(text_input,llm,crosscoder):
    feat_int_SHH = feature_interactions_mlp(text_input,llm,crosscoder,block=0,num_datapoints=1)
    #What I'm doing here is norming over the max feature dimension
    #So that indexing a given feature will say how much it contributes
    #to off dominant features
    feat_int_SH=torch.norm(feat_int_SHH,dim=1)
    
    return feat_int_SH

def get_acts_enc(text_input,llm,crosscoder):
    acts_enc=get_activations(text_input,llm,crosscoder)[0]
    return acts_enc

def get_total_mlp_acts(text_input,llm,crosscoder):
    preacts=get_preacts_mlp(text_input,llm,crosscoder)
    num_features=preacts.shape[-1]
    acts_total=preacts.sum(dim=-1).mean(dim=0)
    #hack, but just repeat it
    acts_total=torch.stack([acts_total for f in range(num_features)],dim=-1)

    return acts_total

def get_token_counts(dataset,llm,crosscoder,num_datapoints=100):
    vocab_size = llm.tokenizer.vocab_size
    W_dec_HXD = crosscoder.W_dec_HXD
    num_features = W_dec_HXD.shape[0]
    token_abs_values = torch.zeros((vocab_size, num_features))
    token_counts=torch.zeros((vocab_size, num_features))
    token_counts_nonzero=torch.zeros((vocab_size, num_features))

    for i in tqdm(range(num_datapoints)):
        text_input=dataset[i]["text"]
        text_input_tokens=torch.tensor(llm.tokenizer.encode(text_input))[:128]
        
        
        feat_int_SH=get_activations(text_input,llm,crosscoder)[0]
        #how to map tokens processed to the token indices?
        #token_abs_values[text_input_tokens.long()]+=feat_int_SH.abs(), below is the same thing but accounts for duplicate tokens per story
        token_abs_values.index_add_(0,text_input_tokens.view(-1).long(),feat_int_SH.abs().to(token_abs_values.dtype))
        #token_counts[text_input_tokens.long()]+=torch.ones_like(feat_int_SH)*torch.nonzero(feat_int_SH)[0]
        mask = feat_int_SH.ne(0).to(token_counts.dtype)   # (S, F) → 0/1 counts
        token_counts.index_add_(0, text_input_tokens.view(-1), mask)
        #mask_nonzero = feat_int_SH.ne(0).to(token_counts_nonzero.dtype)   # (S, F) → 0/1 counts
        

        


    # Return exact memory footprint in bytes
    #size_bytes= token_counts.element_size() * token_counts.numel()
    #print(f"Token counts tensor size: {size_bytes / (1024*1024):.2f} MB")

    return token_abs_values, token_counts

def get_ipr(tok_feats_tensor,r=2):
    tok_feats_squares=(tok_feats_tensor**(2*r)).sum(dim=0)
    tok_feats_sum=((tok_feats_tensor**2).sum(dim=0))**(r/2)
    ipr=tok_feats_squares/tok_feats_sum
    
    return ipr

def get_gini(tok_feats_tensor):
    x = tok_feats_tensor
    T = x.shape[0]
    # sort each feature's values over the T dimension
    sorted_x, _ = torch.sort(x, dim=0)
    # build an index 1…T for the Gini formula
    idx = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(T, 1)
    # Gini coefficient per feature: sum((2*i - T - 1) * x_i) / (T * sum(x))
    gini = ((2 * idx - T - 1) * sorted_x).sum(dim=0) / (T * sorted_x.sum(dim=0) + 1e-8)
    
    return gini

def get_cdf(tok_feats_tensor):
    cdf=torch.cumsum(tok_feats_tensor,dim=0)/tok_feats_tensor.sum(dim=0)
    return cdf

def get_entropy(tok_feats_tensor):
    entropy=(-tok_feats_tensor*torch.log(tok_feats_tensor+1e-8)).sum(dim=0)
    return entropy

# def ablation_mlp(input,llm,crosscoder,block=0):
#     llm.eval()
#     crosscoder.eval()
#     tokens = llm.to_tokens(input)[:,:128]
#     W_in=llm.blocks[block].mlp.W_in
#     b_in=llm.blocks[block].mlp.b_in
#     W_out=llm.blocks[block].mlp.W_out
#     b_out=llm.blocks[block].mlp.b_out
    
#     loss, cache = llm.run_with_cache(tokens, names_filter=hookpoints, return_type="loss")
#     activations_BSLD = torch.stack([cache[name] for name in hookpoints], dim=2)

#     # Ensure all relevant tensors are on the same device
#     device = activations_BSLD.device
#     W_in = W_in.to(device)
#     b_in = b_in.to(device)
#     W_out = W_out.to(device)
#     b_out = b_out.to(device)
    
#     # add model dim 
#     activations_BSXD = torch.unsqueeze(activations_BSLD, dim=2)
#     # remove sequence dim (I'm considering each token in the sequence as a batch)
#     activations_SXD = einops.rearrange(activations_BSXD, "b s m l d -> (b s) m l d")
#     train_res = crosscoder.forward_train(activations_SXD)
#     reconstructed_acts_BXD = train_res.output_BXD

#     # reorder again to remove model dim and add sequence dim
#     reconstructed_acts_BSLD = einops.rearrange(reconstructed_acts_BXD, "(b s) m l d -> b s m l d", b=1)
#     reconstructed_acts_BSLD = reconstructed_acts_BSLD.squeeze(2)

#     print(f'reconstructed_acts_BSLD.shape: {reconstructed_acts_BSLD.shape}')
#     enc_acts,raw_acts=get_activations(input,llm,crosscoder)
    
#     #I think in my rec_acts, I have a model dim, but in this setup that's a batch dim.
#     rec_acts_rearranged=einops.rearrange(rec_acts, "s m l d -> m s l d")
#     mlp_preacts=get_preacts_nocontract(enc_acts,crosscoder.W_dec_HXD,crosscoder.b_dec_XD,llm,block=block,bias=True)
#     mlp_preacts_summed=mlp_preacts.sum(dim=-1)
#     #now you want to see what difference it makes to drop the non-dominant features
#     #mlp_preacts_summed_sorted,indices=torch.sort(mlp_preacts_summed,dim=-1,descending=True)
#     mlp_preacts_max=torch.max(mlp_preacts,dim=-1).values
#     mlp_preacts_nonmax=mlp_preacts_summed-mlp_preacts_max
#     exact_preacts=(activations_BSLD[:, :, 4*block+3, :]@W_in+b_in).squeeze(0)
    
#     print(f'mlp preacts summed.shape: {mlp_preacts_summed.shape}')
    
#     def propagate_preacts(preacts,resid_mid,block=block):
#         mlp_post=nn.GELU()(preacts)
#         mlp_out=einops.einsum(W_out,mlp_post,"d_mlp d_model, batch d_mlp -> batch d_model")+b_out
#         resid_post=resid_mid+mlp_out
#         return resid_post
    
#     prop_max_resid=propagate_preacts(mlp_preacts_max,activations_BSLD[:, :, 4*block+2, :],block=block)
#     prop_nonmax_resid=propagate_preacts(mlp_preacts_nonmax,activations_BSLD[:, :, 4*block+2, :],block=block)
#     prop_sum_resid=propagate_preacts(mlp_preacts_summed,activations_BSLD[:, :, 4*block+2, :],block=block)
#     prop_exact=propagate_preacts(exact_preacts,activations_BSLD[:, :, 4*block+2, :],block=block)

#     #print(f'prop exact loss: {nn.MSELoss()(prop_exact,activations_BSLD[:, :, 4*block+4, :]).item()}')
#     #print(f'prop sum loss: {nn.MSELoss()(prop_sum_resid,activations_BSLD[:, :, 4*block+4, :]).item()}')
#     #print(f'prop max loss: {nn.MSELoss()(prop_max_resid,activations_BSLD[:, :, 4*block+4, :]).item()}')
#     #print(f'prop nonmax loss: {nn.MSELoss()(prop_nonmax_resid,activations_BSLD[:, :, 4*block+4, :]).item()}')
    
#     #raise Exception("Stop here")
    
    
    
    
    
#     #let's calculate the relative loss:    
    
    
    
#     # patch final layer activations into model
    
    
    
#     # def patch_fn(resid_post,acts, hook):
#     #     # extract final layer activations
#     #     return resid_post

#     def make_patch_fn(resid_to_patch_in):
#         # hook signature is (activations, hook_name) → returns the value to use instead
#         def patch_fn(activations, hook):
#             # ignore the model's own activations, return your precomputed residual
#             return resid_to_patch_in
#         return patch_fn
    
    
#     names=['exact','all features','max only','non-max only','zeroes']
#     resids=[prop_exact,prop_sum_resid,prop_max_resid,prop_nonmax_resid,torch.zeros_like(prop_exact)]
#     patched_losses=[]
#     for name,resid in zip(names,resids):
#         patched_loss = llm.run_with_hooks(
#             tokens,
#             return_type="loss",
#             fwd_hooks=[(f"blocks.{block}.hook_resid_post", make_patch_fn(resid))]
#         )
    
#         print(f'{name}: unmodified loss: {loss.item()}, patched loss: {patched_loss.item()}')
#         patched_losses.append(patched_loss.item())
    
#     return loss.item(), patched_losses

def ablation_mlp(input,llm,crosscoder,block=0):
	llm.eval()
	crosscoder.eval()
	tokens = llm.to_tokens(input)[:,:128]
	W_in=llm.blocks[block].mlp.W_in
	b_in=llm.blocks[block].mlp.b_in
	W_out=llm.blocks[block].mlp.W_out
	b_out=llm.blocks[block].mlp.b_out
	
	loss, cache = llm.run_with_cache(tokens, names_filter=hookpoints, return_type="loss")
	activations_BSLD = torch.stack([cache[name] for name in hookpoints], dim=2)
	
	
	
	
	
	
	# add model dim 
	activations_BSXD = torch.unsqueeze(activations_BSLD, dim=2)
	# remove sequence dim (I'm considering each token in the sequence as a batch)
	activations_SXD = einops.rearrange(activations_BSXD, "b s m l d -> (b s) m l d")
	train_res = crosscoder.forward_train(activations_SXD)
	reconstructed_acts_BXD = train_res.output_BXD
	decoded_acts_BD = crosscoder._encode_BH(activations_SXD)

	# reorder again to remove model dim and add sequence dim
	reconstructed_acts_BSLD = einops.rearrange(reconstructed_acts_BXD, "(b s) m l d -> b s m l d", b=1)
	reconstructed_acts_BSLD = reconstructed_acts_BSLD.squeeze(2)

	#enc_acts,raw_acts=get_activations(input,llm,crosscoder)
	
	
	
	#I think in my rec_acts, I have a model dim, but in this setup that's a batch dim.
	#rec_acts_rearranged=einops.rearrange(rec_acts, "s m l d -> m s l d")
	
	
	mlp_preacts=get_preacts_nocontract(decoded_acts_BD,crosscoder.W_dec_HXD,crosscoder.b_dec_XD,llm,block=block,bias=True)
	mlp_preacts_summed=mlp_preacts.sum(dim=-1)
	#now you want to see what difference it makes to drop the non-dominant features
	#mlp_preacts_summed_sorted,indices=torch.sort(mlp_preacts_summed,dim=-1,descending=True)
	mlp_preacts_max=torch.max(mlp_preacts,dim=-1).values
	mlp_preacts_nonmax=mlp_preacts_summed-mlp_preacts_max
	exact_preacts=(activations_BSLD[:, :, 4*block+3, :]@W_in+b_in).squeeze(0)
	
	#print(f'mlp preacts summed.shape: {mlp_preacts_summed.shape}')
	
	
	prop_max_resid=propagate_preacts(mlp_preacts_max,reconstructed_acts_BSLD[:, :, 4*block+2, :],W_out,b_out,block=block)
	prop_nonmax_resid=propagate_preacts(mlp_preacts_nonmax,reconstructed_acts_BSLD[:, :, 4*block+2, :],W_out,b_out,block=block)
	prop_sum_resid=propagate_preacts(mlp_preacts_summed,reconstructed_acts_BSLD[:, :, 4*block+2, :],W_out,b_out,block=block)
	prop_exact=propagate_preacts(exact_preacts,activations_BSLD[:, :, 4*block+2, :],W_out,b_out,block=block)

	#print(f'prop exact loss: {nn.MSELoss()(prop_exact,activations_BSLD[:, :, 4*block+4, :]).item()}')
	#print(f'prop sum loss: {nn.MSELoss()(prop_sum_resid,activations_BSLD[:, :, 4*block+4, :]).item()}')
	#print(f'prop max loss: {nn.MSELoss()(prop_max_resid,activations_BSLD[:, :, 4*block+4, :]).item()}')
	#print(f'prop nonmax loss: {nn.MSELoss()(prop_nonmax_resid,activations_BSLD[:, :, 4*block+4, :]).item()}')
	
	#raise Exception("Stop here")
	
	
	
	
	
	#let's calculate the relative loss:    
	
	
	
	# patch final layer activations into model
	
	
	
	# def patch_fn(resid_post,acts, hook):
	#     # extract final layer activations
	#     return resid_post



	
	names=['original acts','all features','max only','non-max only','zeroes']
	resids=[prop_exact,prop_sum_resid,prop_max_resid,prop_nonmax_resid,torch.zeros_like(prop_exact)]
	patched_losses=[]
	for name,resid in zip(names,resids):
		patched_loss = llm.run_with_hooks(
			tokens,
			return_type="loss",
			fwd_hooks=[(f"blocks.{block}.hook_resid_post", make_patch_fn(resid))]
		)
	
		#print(f'{name}: unmodified loss: {loss.item()}, patched loss: {patched_loss.item()}')
		patched_losses.append(patched_loss.item())
	names.append('model')
	patched_losses.append(loss.item())
	return loss.item(), patched_losses,names
    
def make_ablation_plot(dataset,llm,crosscoder,crosscoder_unpenalized,crosscoder_df,block=3,num_datapoints=10):
	
	loss_rec_vals_list=[]
	patched_loss_list=[]
	

	loss_rec_vals_list_up=[]
	patched_loss_list_up=[]
	

	loss_rec_vals_list_df=[]
	patched_loss_list_df=[]
	

	for i in tqdm(range(num_datapoints)):
		loss,patched_loss,names=ablation_mlp(dataset[i]["text"],llm,crosscoder,block=block)
		loss_rec_vals=1-((np.array(patched_loss)-loss)/(np.array(patched_loss[names.index('zeroes')])-loss))
		loss_rec_vals_list.append(loss_rec_vals)
		patched_loss_list.append(np.array(patched_loss))
	

	
		loss_up,patched_loss_up,names_up=ablation_mlp(dataset[i]["text"],llm,crosscoder_unpenalized,block=block)
		loss_rec_vals_up=1-((np.array(patched_loss_up)-loss_up)/(np.array(patched_loss_up[names_up.index('zeroes')])-loss_up))
		loss_rec_vals_list_up.append(loss_rec_vals_up)
		patched_loss_list_up.append(np.array(patched_loss_up))
		

		loss_df,patched_loss_df,names_df=ablation_mlp(dataset[i]["text"],llm,crosscoder_df,block=block)
		loss_rec_vals_df=1-((np.array(patched_loss_df)-loss_df)/(np.array(patched_loss_df[names_df.index('zeroes')])-loss_df))
		loss_rec_vals_list_df.append(loss_rec_vals_df)
		patched_loss_list_df.append(np.array(patched_loss_df))
	
		#print(f'patched loss list penalized: {np.array(patched_loss_list).mean(axis=0)}')
		#print(f'patched loss list unpenalized: {np.array(patched_loss_list_up).mean(axis=0)}')
		#print(f'patched loss list df: {np.array(patched_loss_list_df).mean(axis=0)}')
		
	
	fig=make_subplots(rows=1,cols=1)
	loss_rec_vals_list=np.array(loss_rec_vals_list).mean(axis=0)
	loss_rec_vals_list_std=np.array(loss_rec_vals_list).std(axis=0)
	loss_rec_vals_list_up=np.array(loss_rec_vals_list_up).mean(axis=0)
	loss_rec_vals_list_up_std=np.array(loss_rec_vals_list_up).std(axis=0)
	loss_rec_vals_list_df=np.array(loss_rec_vals_list_df).mean(axis=0)
	loss_rec_vals_list_df_std=np.array(loss_rec_vals_list_df).std(axis=0)
	for i in range(len(loss_rec_vals_list)-2):
		series=np.array([loss_rec_vals_list[i],loss_rec_vals_list_up[i],loss_rec_vals_list_df[i]])
		series_std=np.array([loss_rec_vals_list_std[i],loss_rec_vals_list_up_std[i],loss_rec_vals_list_df_std[i]])
		
		
		fig.add_trace(go.Bar(
			x=['λ=0','λ=200','λ=1000'],
			y=series,
			name=names[i],
			text=[f'{val:.2f}' for val in series],
			textposition='outside',
			textfont=dict(size=12),
			error_y=dict(
				type='data',
				array=series_std,
				visible=True,
				thickness=1.5,
				width=3
			),
		))
		
	fig.update_layout(
		#title='Recovered loss',
		xaxis_title='Crosscoder',
		yaxis_title='Fidelity',
		uniformtext_minsize=10,
		uniformtext_mode='hide'
	)
	
	return fig

def main():
    

    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")

    dataset = dataset.filter(lambda x: x['is_training'] == True)

    # for i in range(5):
    #     print(dataset[i]["text"])

    # sys.exit()

    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )

    wandb_run_name = "1k68kpv5"  # example – adjust as needed, base XC, l=1000
    #wandb_run_name = "bn1xtudv" #l=2000, bias=True, base XC
    wandb_run_name = "ckubmeg1" #l=1000, bias=True, DF XC
    #wandb_run_name='ckubmeg1' #l=1000, bias=True, DF XC
    wandb_run_name_unpenalized='86u64trx' #l=0, bias=True, base XC
    #wandb_run_name='v7128kc4' #l=1000, mlp_bias=True, DF XC (for sure)
    wandb_run_name_200='7avbfdww'

    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name, "../../.wandb_artifacts", DEVICE
    )

    crosscoder_unpenalized = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name_unpenalized, "../../.wandb_artifacts", DEVICE
    )

    crosscoder_200 = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name_200, "../../.wandb_artifacts", DEVICE
    )

    #get_token_counts(dataset,llm,crosscoder)
    # samples=100
    # tot_abs, counts = get_token_counts(dataset,llm,crosscoder_unpenalized,num_datapoints=samples)
    # tot_abs_nonzero=tot_abs/counts.clamp(min=1)
    # pickle.dump(tot_abs_nonzero,open(f'feature_interactions/token_feats_abs_nonzeromean_{wandb_run_name_unpenalized}_samples_{samples}.pkl','wb'))
    # sys.exit()
    
    
    


    

    #crosscoder_unpenalized = load_crosscoder_from_wandb(
    #    "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name_unpenalized, "../../.wandb_artifacts", DEVICE
    #)
        
    #get the unpenalized activations
    #unpenalized_abs,unpenalized_counts=get_token_counts(dataset,llm,crosscoder_unpenalized,num_datapoints=300)

    

    #blated_losses=ablation_mlp(dataset[0]["text"],llm,crosscoder,block=3)
    #ablated_losses_unpenalized=ablation_mlp(dataset[0]["text"],llm,crosscoder_unpenalized,block=3)

    fig=make_ablation_plot(dataset,llm,crosscoder,crosscoder_200,crosscoder_unpenalized,block=3,num_datapoints=10)
    fig.write_image('ablation.pdf',engine='kaleido',width=2000,height=600)
    sys.exit()


    
    #print(f'ablated_losses: {ablated_losses}')
    #print(f'ablated_losses_unpenalized: {ablated_losses_unpenalized}')
    sys.exit("huzzah")
    
    #raise Exception("Stop here")
    
    

    data_folder_path = os.getcwd() + "/data/features"
    filename = None
    #feature_stats_10_2025-04-23_22-29-32.pkl #v7128kc4
    #feature_stats_10_2025-04-23_21-57-39.pkl #1k68kpv5
    #'feature_stats_10_2025-04-23_21-32-05.pkl'#86u64trx. 
    
    #'feature_stats_1_2025-04-21_14-51-45.pkl'#'feature_stats_1_2025-04-18_13-38-29.pkl'  # or set to an existing pickle if desired
    #/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/data/features/feature_stats_10_2025-04-23_21-32-05.pkl
    num_samples=10
    if filename is None:
        from sleepers.autointerp.util.activation_util import get_feature_stats_rapid

        tensor_data_dict = get_feature_stats_rapid(dataset, llm, crosscoder, num_samples=num_samples)
        #save_dict(tensor_data_dict, f"feature_stats_{num_samples}")
        save_dict(tensor_data_dict, f"feature_stats_b.pkl", path=data_folder_path)
    else:
        tensor_data_dict = load_dict(f"{data_folder_path}/{filename}")

    test_ft_int=feature_interactions_mlp(dataset[0]["text"],llm,crosscoder)

    

    # feature_interactions = feature_interactions_sum(1, 20,dataset,llm,crosscoder)
    
    

    # Print the top-k largest entries in feature_interactions
    # k = 10
    # flat = feature_interactions.flatten()
    # # get indices of the top k values
    # topk_idx = np.argsort(flat)[::-1][:k]
    # rows, cols = np.unravel_index(topk_idx, feature_interactions.shape)
    # print(f"Top {k} feature interactions (feature_i, feature_j, value):")
    # for r, c, idx_flat in zip(rows, cols, topk_idx):
    #     val = flat[idx_flat]
    #     print(f"Feature {r} & Feature {c}: {val:.6f}")
    
    # raise Exception("Stop here")
    #layer=1
    #num_datapoints=200
    #test_ft_int=feature_interactions_alltokens(layer,num_datapoints,dataset,llm,crosscoder)

    #feat_int_dashboard(tensor_data_dict,test_ft_int, dataset, llm, crosscoder, [get_acts_enc,get_preacts_mlp], top_k=5,number_of_examples=5)
    #raise Exception("Tezting, init")
    build_dashboard(tensor_data_dict, dataset, llm, crosscoder, [get_acts_enc,get_preacts_mlp], top_k=12,number_of_examples=4)
    

if __name__ == "__main__":
    main()
