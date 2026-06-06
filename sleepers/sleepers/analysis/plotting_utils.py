"""
Interactive plotting utilities for feature analysis visualizations.

This module provides interactive HTML-based visualizations for exploring
feature activations in language models with crosscoders.
"""

import torch
import pandas as pd
from IPython.display import HTML
import json
from sleepers.analysis.analysis_utils import get_activations


def load_feature_explanations(wandb_run_name, autointerp_data_path=None):
    """
    Load feature explanations from CSV file.
    
    Args:
        wandb_run_name (str): The wandb run name to load explanations for
        autointerp_data_path (str, optional): Path to autointerp data directory.
            If None, uses default path.
    
    Returns:
        dict: Mapping from feature_id to explanation text
    """
    if autointerp_data_path is None:
        autointerp_data_path = ("/Users/dmitrymanning-coe/Documents/Research/"
                               "compact_proofs/code/post_fork/crosscoders-feature-interactions/"
                               "sleepers/sleepers/autointerp/autointerp_data")
    
    csv_path = f"{autointerp_data_path}/explanations_{wandb_run_name}_withhate.csv"
    print(f'csv path: {csv_path}')
    
    try:
        df = pd.read_csv(csv_path)
        explanations = dict(zip(df['feature_id'], df['explanation']))
        print(f"✓ Loaded {len(explanations)} explanations from {csv_path}")
        return explanations
    except FileNotFoundError:
        print(f"⚠️ No explanations found for run '{wandb_run_name}' at {csv_path}")
        return {}
    except Exception as e:
        print(f"⚠️ Error loading explanations: {e}")
        return {}


def create_interactive_story_visualization(story_text, llm, crosscoder, 
                                         wandb_run_name=None, top_k=10, 
                                         explanations=None, autointerp_data_path=None):
    """
    Create an interactive visualization with story text and dynamic feature table.
    
    This creates a two-panel layout:
    - Top: Dynamic table showing top features for hovered token
    - Bottom: Story text with tokens colored by activation strength
    
    Args:
        story_text (str): The story text to analyze
        llm: The language model (with tokenizer)
        crosscoder: The crosscoder model for feature extraction
        wandb_run_name (str, optional): Run name for loading explanations
        top_k (int): Number of top features to show in table (default: 10)
        explanations (dict, optional): Pre-loaded explanations dict. If None and 
            wandb_run_name provided, will attempt to load from CSV.
        autointerp_data_path (str, optional): Path to autointerp data directory
    
    Returns:
        IPython.display.HTML: Interactive HTML widget
    """
    
    # Get feature activations for the story
    print(f"Analyzing story: {story_text[:100]}...")
    feature_activations, _ = get_activations(story_text, llm, crosscoder)
    print(f"✓ Extracted feature activations: {feature_activations.shape}")
    
    # Load explanations if needed
    if explanations is None and wandb_run_name is not None:
        explanations = load_feature_explanations(wandb_run_name, autointerp_data_path)
    elif explanations is None:
        explanations = {}
        print("⚠️ No explanations provided - using empty dict")
    
    # Get tokens and decode
    tokens = llm.tokenizer.encode(story_text)
    
    tokens = tokens[:feature_activations.shape[0]]  # Match activation length
    decoded_tokens = [llm.tokenizer.decode([token]) for token in tokens]
    
    print(f"✓ Tokenized story: {len(decoded_tokens)} tokens")
    
    # Get top k features per token
    top_values, top_features = torch.topk(feature_activations.abs(), k=top_k, dim=1)
    
    # Prepare data for JavaScript
    token_data = []
    for i, (token_features, token_values) in enumerate(zip(top_features, top_values)):
        token_info = {
            'token': decoded_tokens[i],
            'token_id': i,
            'max_activation': feature_activations[i].abs().max().item(),
            'features': []
        }
        
        for j, (feat_idx, feat_val) in enumerate(zip(token_features, token_values)):
            feat_id = feat_idx.item()
            token_info['features'].append({
                'rank': j + 1,
                'feature_id': feat_id,
                'activation': feat_val.item(),
                'explanation': explanations.get(feat_id, "No explanation available")
            })
        
        token_data.append(token_info)
    
    # Find global max for color normalization
    global_max = feature_activations.abs().max().item()
    
    # Create run info for display
    run_info = f" (Run: {wandb_run_name})" if wandb_run_name else ""
    
    # Generate HTML with embedded JavaScript
    html_content = f"""
    <div id="interactive-story-container" style="font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto;">
        
        <!-- Header -->
        <div style="margin-bottom: 20px; text-align: center;">
            <h2 style="color: #212529; margin-bottom: 5px;">Interactive Feature Analysis{run_info}</h2>
            <p style="color: #6c757d; margin-top: 5px;">Hover over tokens to explore their top {top_k} activating features</p>
        </div>
        
        <!-- Feature Table at Top -->
        <div id="feature-table-container" style="margin-bottom: 30px; border: 2px solid #ccc; border-radius: 8px; padding: 20px; background: #f8f9fa;">
            <h3 id="table-title" style="margin-top: 0; color: #212529; font-weight: bold;">Hover over a token below to see its top {top_k} features</h3>
            <div id="feature-table" style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: #343a40;">
                            <th style="border: 1px solid #adb5bd; padding: 12px; text-align: center; color: white; font-weight: bold;">Rank</th>
                            <th style="border: 1px solid #adb5bd; padding: 12px; text-align: center; color: white; font-weight: bold;">Feature</th>
                            <th style="border: 1px solid #adb5bd; padding: 12px; text-align: center; color: white; font-weight: bold;">Activation</th>
                            <th style="border: 1px solid #adb5bd; padding: 12px; color: white; font-weight: bold;">Explanation</th>
                        </tr>
                    </thead>
                    <tbody id="feature-table-body">
                        <tr><td colspan="4" style="text-align: center; padding: 20px; color: #495057; font-size: 16px;">Hover over tokens below...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Story Text at Bottom -->
        <div id="story-container" style="border: 2px solid #ccc; border-radius: 8px; padding: 20px; background: white; line-height: 1.8; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #212529; font-weight: bold;">Story Text (hover over tokens):</h3>
            <div id="story-text" style="font-size: 16px;">
                <!-- Tokens will be inserted here by JavaScript -->
            </div>
        </div>
        
        <!-- Footer with stats -->
        <div style="margin-top: 20px; text-align: center; color: #6c757d; font-size: 14px;">
            <p>Story: {len(decoded_tokens)} tokens | Features: {feature_activations.shape[1]} total | Max activation: {global_max:.4f}</p>
        </div>
        
    </div>

    <script>
    (function() {{
        // Token data
        const tokenData = {json.dumps(token_data)};
        const globalMax = {global_max};
        const topK = {top_k};
        
        console.log('Interactive story visualization loaded');
        console.log('Token data:', tokenData.length, 'tokens');
        console.log('Global max activation:', globalMax);
        
        // TransformerLens-style red gradient color function
        function getRedGradientColor(activation, maxVal) {{
            const normalized = Math.min(activation / (maxVal + 1e-8), 1.0);
            
            // Red gradient: from white (0) to deep red (1)
            const red = Math.floor(255);
            const green = Math.floor(255 * (1 - normalized * 0.85));
            const blue = Math.floor(255 * (1 - normalized * 0.85));
            
            return `rgb(${{red}}, ${{green}}, ${{blue}})`;
        }}
        
        // Get text color based on background intensity
        function getTextColor(activation, maxVal) {{
            const normalized = Math.min(activation / (maxVal + 1e-8), 1.0);
            return normalized > 0.6 ? 'white' : '#212529';
        }}
        
        // Initialize story text with clickable tokens
        function initializeStory() {{
            console.log('Initializing story...');
            const storyContainer = document.getElementById('story-text');
            
            if (!storyContainer) {{
                console.error('Story container not found!');
                return;
            }}
            
            let storyHTML = '';
            
            tokenData.forEach((tokenInfo, index) => {{
                const backgroundColor = getRedGradientColor(tokenInfo.max_activation, globalMax);
                const textColor = getTextColor(tokenInfo.max_activation, globalMax);
                const borderColor = getRedGradientColor(tokenInfo.max_activation * 1.2, globalMax);
                
                storyHTML += `<span 
                    class="token" 
                    data-token-id="${{index}}"
                    style="
                        background-color: ${{backgroundColor}};
                        color: ${{textColor}};
                        border: 2px solid ${{borderColor}};
                        border-radius: 4px;
                        padding: 4px 6px;
                        margin: 2px;
                        cursor: pointer;
                        display: inline-block;
                        transition: all 0.2s ease;
                        font-weight: 500;
                        text-shadow: ${{textColor === 'white' ? '1px 1px 2px rgba(0,0,0,0.7)' : 'none'}};
                    "
                    onmouseover="showFeatures(${{index}})"
                    onmouseout="clearFeatures()"
                >${{tokenInfo.token}}</span>`;
            }});
            
            storyContainer.innerHTML = storyHTML;
            console.log('Story initialized with', tokenData.length, 'tokens');
        }}
        
        // Show features for a token
        window.showFeatures = function(tokenIndex) {{
            const tokenInfo = tokenData[tokenIndex];
            const tableTitle = document.getElementById('table-title');
            const tableBody = document.getElementById('feature-table-body');
            
            if (!tokenInfo || !tableTitle || !tableBody) {{
                console.error('Missing elements for showFeatures');
                return;
            }}
            
            // Update title
            tableTitle.innerHTML = `Top ${{topK}} features for token "<strong style="color: #dc3545;">${{tokenInfo.token}}</strong>" (position ${{tokenIndex}})`;
            
            // Update table
            let tableHTML = '';
            tokenInfo.features.forEach((feature, idx) => {{
                const rowBg = idx % 2 === 0 ? '#f8f9fa' : 'white';
                tableHTML += `
                    <tr style="background-color: ${{rowBg}};">
                        <td style="border: 1px solid #dee2e6; padding: 12px; text-align: center; font-weight: bold; color: #212529; font-size: 14px;">${{feature.rank}}</td>
                        <td style="border: 1px solid #dee2e6; padding: 12px; text-align: center; font-weight: bold; color: #dc3545; font-size: 14px;">F${{feature.feature_id}}</td>
                        <td style="border: 1px solid #dee2e6; padding: 12px; text-align: center; font-family: 'Courier New', monospace; color: #212529; font-weight: bold; font-size: 14px;">${{feature.activation.toFixed(4)}}</td>
                        <td style="border: 1px solid #dee2e6; padding: 12px; color: #212529; font-size: 14px; line-height: 1.4;">${{feature.explanation}}</td>
                    </tr>
                `;
            }});
            
            tableBody.innerHTML = tableHTML;
            
            // Highlight the hovered token
            document.querySelectorAll('.token').forEach(token => {{
                token.style.transform = 'scale(1)';
                token.style.boxShadow = 'none';
                token.style.zIndex = '1';
            }});
            
            const hoveredToken = document.querySelector(`[data-token-id="${{tokenIndex}}"]`);
            if (hoveredToken) {{
                hoveredToken.style.transform = 'scale(1.15)';
                hoveredToken.style.boxShadow = '0 4px 12px rgba(220, 53, 69, 0.4)';
                hoveredToken.style.zIndex = '10';
            }}
        }};
        
        // Clear features
        window.clearFeatures = function() {{
            document.querySelectorAll('.token').forEach(token => {{
                token.style.transform = 'scale(1)';
                token.style.boxShadow = 'none';
                token.style.zIndex = '1';
            }});
        }};
        
        // Initialize immediately and on DOM ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initializeStory);
        }} else {{
            initializeStory();
        }}
        
        // Also try after a short delay to ensure everything is loaded
        setTimeout(initializeStory, 100);
    }})();
    </script>
    """
    
    print("✓ Generated interactive visualization")
    return HTML(html_content)


# Main streamlined function with all logic built-in
def plot_story_features(llm, crosscoder, wandb_run_name, story_text, top_k=10):
    """
    One-line function to create interactive story feature visualization.
    Handles all data loading and setup internally.
    
    Args:
        llm: The language model (with tokenizer)
        crosscoder: The crosscoder model for feature extraction
        wandb_run_name (str): Run name for loading explanations
        story_index (int): Index of story to analyze from dataset (default: 0)
        top_k (int): Number of top features to show (default: 10)
    
    Returns:
        IPython.display.HTML: Interactive HTML widget
    """
    # try:
    #     # Load dataset if needed
    #     print("Loading TinyStories dataset...")
    #     from datasets import load_dataset
    #     dataset = load_dataset('mars-jason-25/tiny_stories_instruct_sleeper_data', split='train')
    #     dataset = dataset.filter(lambda x: x['is_training'] == True)
    #     print(f"✓ Loaded dataset with {len(dataset)} stories")
        
    # Get the story
    print(f"✓ Selected story: {story_text[:100]}...")
        
    
    # Create the visualization
    return create_interactive_story_visualization(
        story_text=story_text,
        llm=llm,
        crosscoder=crosscoder,
        wandb_run_name=wandb_run_name,
        top_k=top_k
    )


# Alternative function that takes a custom story
def plot_custom_story_features(story_text, llm, crosscoder, wandb_run_name, top_k=10):
    """
    Simple wrapper for creating visualization with custom story text.
    
    Args:
        story_text (str): The story text to analyze
        llm: The language model (with tokenizer)
        crosscoder: The crosscoder model for feature extraction
        wandb_run_name (str): Run name for loading explanations
        top_k (int): Number of top features to show (default: 10)
    
    Returns:
        IPython.display.HTML: Interactive HTML widget
    """
    return create_interactive_story_visualization(
        story_text=story_text,
        llm=llm,
        crosscoder=crosscoder,
        wandb_run_name=wandb_run_name,
        top_k=top_k
    )