import torch
import torch.nn as nn
import numpy as np
from typing import Any
import matplotlib.colors as mcolors
from IPython.display import HTML
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML
from matplotlib import pyplot as plt

def compute_cosine_similarities(features_1: torch.Tensor, features_2: torch.Tensor) -> np.ndarray[Any, np.dtype[np.float64]]:
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cosine_sims = []
    for i in range(features_1.shape[0]):
        cosine_sims.append(cos(features_1[i], features_2[i]).to('cpu').detach().numpy())
    return np.array(cosine_sims)

# def display_text_with_highlighting(tokens, tokenizer, values, cmap=None, vmin=None, vmax=None, transparent_test=None):
#     cmap = plt.get_cmap('viridis') if cmap is None else cmap
#     vmin = vmin if vmin is not None else values.min()
#     vmax = vmax if vmax is not None else values.max()
#     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#     html = "<div style='font-family: monospace; line-height: 1.5; margin-bottom: 15px;'>"
    
#     for i, token_id in enumerate(tokens):
#         token_text = tokenizer.decode([token_id])
#         color_rgb = cmap(norm(values[i]))
#         if transparent_test is not None and transparent_test(values[i]):
#             color = 'transparent'
#         else:
#             rgba = f'rgba({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)}, 0.7)'
#             color = rgba
            
#         # Add title attribute for tooltip with the activation value
#         html += f"<span style='background-color: {color}; padding: 2px; margin: 1px; border-radius: 3px; cursor: pointer;' title='Activation: {values[i]:.4f}'>{token_text}</span>"
    
#     html += "</div>"
#     return HTML(html)

def display_feature_activation_visualization(tokenizer, example_texts, example_activations, feature_index, vmin=None, vmax=None):
    """
    Display the feature activation visualization in a Jupyter notebook.
    """
    if vmin is None or vmax is None:
        all_feature_activations = torch.cat([act[:, feature_index] for act in example_activations])
        global_min = float(all_feature_activations.min())
        global_max = float(all_feature_activations.max())
        vmin = vmin if vmin is not None else global_min
        vmax = vmax if vmax is not None else global_max
    
    cmap = mcolors.LinearSegmentedColormap.from_list('feature_activation', 
                                                     ['black', 'blue', 'red'])
    # Generate unified color bar
    gradient_colors = [mcolors.to_hex(cmap(i/100)) for i in range(101)]
    gradient_css = ','.join(gradient_colors)
    colorbar = f'''
    <div style="width:100%; margin:10px 0;">
        <div style="width:100%; height:20px; background: linear-gradient(to right, {gradient_css});"></div>
        <div style="display:flex; justify-content:space-between; margin-top:2px;">
            <span>Low Activation ({vmin:.3f})</span>
            <span>High Activation ({vmax:.3f})</span>
        </div>
    </div>
    '''
    
    # Create all visualizations
    html_visualizations = []
    for text, activations in zip(example_texts, example_activations):
        tokens = tokenizer.encode(text)[:128]
        feature_activations = activations[:len(tokens), feature_index].detach().cpu().numpy()
        html = display_text_with_highlighting(
            tokens,
            tokenizer,
            feature_activations,
            cmap,
            vmin, vmax,
            lambda x: x == 0)
        html_visualizations.append(html.data)
    
    # Combine everything with single title and colorbar at top
    combined_html = f"""
    <div style='display: flex; flex-direction: column; gap: 0px;'>
        <h3>Feature {feature_index} Activation Visualization</h3>
        {colorbar}
        {"".join(html_visualizations)}
    </div>
    """
    return HTML(combined_html)
def display_text_with_highlighting(tokens, tokenizer, values, cmap=None, vmin=None, vmax=None, transparent_test=None):
    cmap = plt.get_cmap('viridis') if cmap is None else cmap
    vmin = vmin if vmin is not None else values.min()
    vmax = vmax if vmax is not None else values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Create HTML with highlighting
    html = "<div style='font-family: monospace; line-height: 1.5; margin-bottom: 15px;'>"

    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        color_rgb = cmap(norm(values[i]))
        if transparent_test is not None and transparent_test(values[i]):
            color = 'transparent'
        else:
            # Convert to rgba with 70% opacity (30% transparent)
            rgba = f'rgba({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)}, 0.7)'
            color = rgba
        html += f"<span style='background-color: {color}; padding: 2px; margin: 1px; border-radius: 3px;'>{token_text}</span>"
    
    html += "</div>"
    return HTML(html)

def display_feature_activation_visualization(tokenizer, example_texts, example_activations, feature_index, vmin=None, vmax=None,int_metric_label=False):
    """
    Display the feature activation visualization in a Jupyter notebook.
    """
    if vmin is None or vmax is None:
        all_feature_activations = torch.cat([act[:, feature_index] for act in example_activations])
        global_min = float(all_feature_activations.min())
        global_max = float(all_feature_activations.max())

        vmin = vmin if vmin is not None else global_min
        vmax = vmax if vmax is not None else global_max

    cmap = mcolors.LinearSegmentedColormap.from_list('feature_activation', 
                                                     ['black', 'blue', 'red'])
    # Generate unified color bar
    gradient_colors = [mcolors.to_hex(cmap(i/100)) for i in range(101)]
    gradient_css = ','.join(gradient_colors)
    colorbar = f'''
    <div style="width:100%; margin:10px 0;">
        <div style="width:100%; height:20px; background: linear-gradient(to right, {gradient_css});"></div>
        <div style="display:flex; justify-content:space-between; margin-top:2px;">
            <span>Low Activation ({vmin:.3f})</span>
            <span>High Activation ({vmax:.3f})</span>
        </div>
    </div>
    '''
    
    # Create all visualizations
    html_visualizations = []
    for text, activations in zip(example_texts, example_activations):
        tokens = tokenizer.encode(text)[:128]
        feature_activations = activations[:len(tokens), feature_index].detach().cpu().numpy()
        
        
        html = display_text_with_highlighting(
            tokens,
            tokenizer,
            feature_activations,
            cmap,
            vmin, vmax,
            lambda x: x == 0)
        html_visualizations.append(html.data)
    
    # Combine everything with single title and colorbar at top
    if int_metric_label:
        label=f'Interaction'
    else:
        label=f'Feature {feature_index}'
    combined_html = f"""
    <div style='display: flex; flex-direction: column; gap: 0px;'>
        <h3>{label} Activation Visualization</h3>
        {colorbar}
        {"".join(html_visualizations)}
    </div>
    """
    return HTML(combined_html)


    