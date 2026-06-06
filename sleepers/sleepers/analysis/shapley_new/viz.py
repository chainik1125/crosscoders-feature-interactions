#!/usr/bin/env python
"""
Dashboard for visualizing Shapley feature interactions.
Adapted from https://github.com/chainik1125/fra/tree/main/fra
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests


def load_feature_explanations_from_csv(crosscoder_name: str) -> Dict[int, str]:
    """Load feature explanations from local CSV file."""
    import pandas as pd
    explanations = {}

    csv_path = f"/root/crosscoders-feature-interactions/sleepers/sleepers/autointerp/autointerp_data/explanations_{crosscoder_name}.csv"

    try:
        df = pd.read_csv(csv_path)
        # Assuming the CSV has columns like 'feature_id' and 'explanation' or similar
        # Adjust column names based on actual CSV structure
        if 'feature_id' in df.columns and 'explanation' in df.columns:
            for _, row in df.iterrows():
                explanations[int(row['feature_id'])] = str(row['explanation'])
        elif 'Feature' in df.columns and 'Explanation' in df.columns:
            for _, row in df.iterrows():
                explanations[int(row['Feature'])] = str(row['Explanation'])
        else:
            # Try to infer columns
            cols = df.columns.tolist()
            if len(cols) >= 2:
                for _, row in df.iterrows():
                    try:
                        explanations[int(row[cols[0]])] = str(row[cols[1]])
                    except:
                        continue
    except Exception as e:
        print(f"Could not load explanations from {csv_path}: {e}")

    return explanations


def process_shapley_tensors(
    pairwise_signed_sum: np.ndarray,
    pairwise_counts: np.ndarray,
    feature_counts: np.ndarray,
    single_indices_sum: np.ndarray,
    pairwise_abs_sum: Optional[np.ndarray] = None,
    use_abs_values: bool = True,
    eps: float = 1e-10,
    top_k: int = 100
) -> Dict[str, Any]:
    """
    Process Shapley accumulation results to extract top feature pairs.

    Args:
        pairwise_signed_sum: [H, H] array of summed pairwise Shapley-Taylor indices
        pairwise_counts: [H, H] array of pairwise co-occurrence counts
        feature_counts: [H] array of feature occurrence counts
        single_indices_sum: [H] array of summed single feature indices
        pairwise_abs_sum: [H, H] array of absolute summed pairwise indices (optional)
        use_abs_values: Whether to use absolute sum for ranking (default: True)
        eps: Minimum threshold for considering a pair active (default: 1e-10)
        top_k: Number of top pairs to return

    Returns:
        Dictionary with top feature pairs and their statistics
    """
    # Get non-zero pairs with masking for near-zero values
    feature_pairs = []
    H = pairwise_signed_sum.shape[0]

    # Decide which values to use for ranking
    if use_abs_values and pairwise_abs_sum is not None:
        ranking_values = pairwise_abs_sum
    else:
        ranking_values = np.abs(pairwise_signed_sum)

    for i in range(H):
        for j in range(i, H):  # Only upper triangle including diagonal
            # Only consider pairs with counts > eps
            if pairwise_counts[i, j] > eps:
                avg_signed = pairwise_signed_sum[i, j] / pairwise_counts[i, j]
                avg_abs = ranking_values[i, j] / pairwise_counts[i, j]

                feature_pairs.append({
                    'feature_1': i,
                    'feature_2': j,
                    'signed_sum': pairwise_signed_sum[i, j],
                    'abs_sum': ranking_values[i, j],
                    'count': pairwise_counts[i, j],
                    'average_signed': avg_signed,
                    'average_abs': avg_abs,
                    'is_self': i == j
                })

    # Sort by absolute average interaction strength
    feature_pairs.sort(key=lambda x: x['average_abs'], reverse=True)

    # Keep top k
    top_pairs = feature_pairs[:top_k]

    # Calculate statistics for single features with masking
    active_features = np.where(feature_counts > eps)[0]
    single_averages = {}
    for feat in active_features:
        if feature_counts[feat] > eps:
            single_averages[feat] = single_indices_sum[feat] / feature_counts[feat]

    # Calculate absolute values for all pairs (for CCDF)
    all_abs_averages = [p['average_abs'] for p in feature_pairs]

    return {
        'top_pairs': top_pairs,
        'all_pairs': feature_pairs,  # Keep all pairs for CCDF
        'total_pairs': len(feature_pairs),
        'single_averages': single_averages,
        'active_features': active_features.tolist(),
        'max_average_abs': top_pairs[0]['average_abs'] if top_pairs else 0,
        'min_average_abs': top_pairs[-1]['average_abs'] if top_pairs else 0,
        'all_abs_averages': all_abs_averages
    }


def create_shapley_dashboard(
    pairwise_signed_sum: np.ndarray,
    pairwise_counts: np.ndarray,
    feature_counts: np.ndarray,
    single_indices_sum: np.ndarray,
    pairwise_abs_sum: Optional[np.ndarray] = None,
    total_samples: int = 0,
    model_name: str = "TinyStories-33M",
    layer_name: str = "blocks.0.hook_resid_pre",
    crosscoder_name: str = "ckubmeg1",
    top_k: int = 50,
    output_path: Optional[str] = None,
    fetch_neuronpedia: bool = False
) -> str:
    """
    Create an interactive dashboard for Shapley feature interactions.

    Args:
        pairwise_signed_sum: [H, H] array of summed pairwise Shapley-Taylor indices
        pairwise_counts: [H, H] array of pairwise co-occurrence counts
        feature_counts: [H] array of feature occurrence counts
        single_indices_sum: [H] array of summed single feature indices
        pairwise_abs_sum: [H, H] array of absolute summed pairwise indices (optional)
        total_samples: Total number of samples processed
        model_name: Name of the model for display
        layer_name: Name of the layer/hook point
        crosscoder_name: Name of the crosscoder (for loading explanations)
        top_k: Number of top pairs to visualize
        output_path: Where to save the dashboard
        fetch_neuronpedia: Whether to fetch feature descriptions from Neuronpedia (deprecated, uses CSV)

    Returns:
        Path to saved dashboard
    """
    # Process tensors to get top pairs
    results = process_shapley_tensors(
        pairwise_signed_sum,
        pairwise_counts,
        feature_counts,
        single_indices_sum,
        pairwise_abs_sum=pairwise_abs_sum,
        use_abs_values=True,
        eps=1e-10,
        top_k=top_k
    )

    top_pairs = results['top_pairs']
    single_averages = results['single_averages']

    # Load feature descriptions from CSV
    feature_descriptions = load_feature_explanations_from_csv(crosscoder_name)

    # Add default descriptions for any missing features
    unique_features = set()
    for pair in top_pairs:
        unique_features.add(pair['feature_1'])
        unique_features.add(pair['feature_2'])

    for feat_id in unique_features:
        if feat_id not in feature_descriptions:
            feature_descriptions[feat_id] = f"Feature {feat_id} - no description"

    # Create visualizations with CCDF
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            f'Top {min(20, len(top_pairs))} Feature Interactions (Abs Shapley-Taylor)',
            'CCDF of Absolute Interaction Strengths',
            'Feature Interaction Heatmap',
            'Single Feature Shapley Values',
            'Self-Interaction vs Cross-Interaction',
            'Interaction Strength vs Frequency',
            'Feature Activity Distribution',
            'Signed vs Absolute Values'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'scatter'}],
            [{'type': 'heatmap'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'scatter'}],
            [{'type': 'histogram'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.15
    )

    # 1. Top interactions bar chart
    top_20 = top_pairs[:20]
    pair_labels = []
    pair_values = []
    pair_counts = []
    pair_colors = []

    for pair in top_20:
        f1 = pair['feature_1']
        f2 = pair['feature_2']
        f1_desc = feature_descriptions.get(f1, f"F{f1}")[:25]
        f2_desc = feature_descriptions.get(f2, f"F{f2}")[:25]

        if pair['is_self']:
            label = f"🔄 {f1_desc}"
            color = 'red'
        else:
            label = f"{f1_desc} ↔ {f2_desc}"
            color = 'blue' if pair['average_signed'] > 0 else 'orange'

        pair_labels.append(label)
        pair_values.append(pair['average_abs'])  # Use absolute average
        pair_counts.append(pair['count'])
        pair_colors.append(color)

    fig.add_trace(
        go.Bar(
            x=pair_values,
            y=pair_labels,
            orientation='h',
            marker_color=pair_colors,
            text=[f"n={c}" for c in pair_counts],
            textposition='auto',
            name='Shapley Interaction',
            hovertemplate="<b>%{y}</b><br>Abs Shapley: %{x:.6f}<br>Count: %{text}<extra></extra>"
        ),
        row=1, col=1
    )

    # 2. CCDF of absolute interaction strengths
    if results.get('all_abs_averages'):
        # Sort values in descending order for CCDF
        sorted_abs = sorted(results['all_abs_averages'], reverse=True)
        n = len(sorted_abs)
        # CCDF: probability of seeing a value >= x
        ccdf_probs = [(i + 1) / n for i in range(n)]

        fig.add_trace(
            go.Scatter(
                x=sorted_abs,
                y=ccdf_probs,
                mode='lines',
                line=dict(color='purple', width=2),
                name='CCDF',
                hovertemplate="Value: %{x:.6f}<br>P(X≥x): %{y:.4f}<extra></extra>"
            ),
            row=1, col=2
        )

        # Add log scale to both axes for better visualization
        fig.update_xaxes(type="log", title_text="Abs Interaction Strength (log)", row=1, col=2)
        fig.update_yaxes(type="log", title_text="P(X ≥ x) (log)", row=1, col=2)

    # 3. Top single feature Shapley values (moved to row 2, col 2)
    top_singles = sorted(single_averages.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    if top_singles:
        single_labels = [feature_descriptions.get(f[0], f"F{f[0]}")[:30] for f in top_singles]
        single_values = [f[1] for f in top_singles]
        single_counts = [feature_counts[f[0]] for f in top_singles]

        fig.add_trace(
            go.Bar(
                x=single_values,
                y=single_labels,
                orientation='h',
                marker_color=['green' if v > 0 else 'purple' for v in single_values],
                text=[f"n={c}" for c in single_counts],
                textposition='auto',
                name='Single Feature',
                hovertemplate="<b>%{y}</b><br>Abs Shapley: %{x:.6f}<br>Count: %{text}<extra></extra>"
            ),
            row=1, col=2
        )

    # 3. Interaction heatmap (top features)
    # Get most active features by total absolute interaction
    feature_activity = {}
    for pair in top_pairs:
        f1 = pair['feature_1']
        f2 = pair['feature_2']
        feature_activity[f1] = feature_activity.get(f1, 0) + pair['average_abs']
        if not pair['is_self']:
            feature_activity[f2] = feature_activity.get(f2, 0) + pair['average_abs']

    top_features = sorted(feature_activity.items(), key=lambda x: x[1], reverse=True)[:25]
    top_feature_ids = [f[0] for f in top_features]

    # Build heatmap matrix
    heatmap_size = len(top_feature_ids)
    heatmap_matrix = np.zeros((heatmap_size, heatmap_size))

    for pair in top_pairs:
        if pair['feature_1'] in top_feature_ids and pair['feature_2'] in top_feature_ids:
            i = top_feature_ids.index(pair['feature_1'])
            j = top_feature_ids.index(pair['feature_2'])
            # Use absolute average for heatmap
            heatmap_matrix[i, j] = pair['average_abs']
            if not pair['is_self']:
                heatmap_matrix[j, i] = pair['average_abs']

    feature_labels = [feature_descriptions.get(f, f"F{f}")[:15] for f in top_feature_ids]

    fig.add_trace(
        go.Heatmap(
            z=heatmap_matrix,
            x=feature_labels,
            y=feature_labels,
            colorscale='Viridis',  # Use Viridis for absolute values
            text=[[f"{val:.3f}" if abs(val) > 0.001 else "" for val in row] for row in heatmap_matrix],
            texttemplate="%{text}",
            textfont={"size": 8},
            showscale=True,
            colorbar=dict(title="Abs Shapley")
        ),
        row=2, col=1
    )

    # 4. Self-interactions vs cross-interactions
    self_interactions = [p for p in top_pairs if p['is_self']][:15]
    cross_interactions = [p for p in top_pairs if not p['is_self']][:15]

    categories = []
    values = []
    colors = []
    texts = []

    for si in self_interactions:
        categories.append("Self")
        values.append(si['average_abs'])
        colors.append('red')
        texts.append(f"F{si['feature_1']}")

    for ci in cross_interactions:
        categories.append("Cross")
        values.append(ci['average_abs'])
        colors.append('blue')  # Always blue since we're using absolute values
        texts.append(f"F{ci['feature_1']}↔F{ci['feature_2']}")

    if categories:
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=texts,
                name='Interaction Type',
                hovertemplate="<b>%{text}</b><br>Type: %{x}<br>Abs Shapley: %{y:.6f}<extra></extra>"
            ),
            row=2, col=2
        )

    # 5. Interaction strength vs frequency scatter
    fig.add_trace(
        go.Scatter(
            x=[p['count'] for p in top_pairs],
            y=[p['average_abs'] for p in top_pairs],
            mode='markers',
            marker=dict(
                size=8,
                color=['red' if p['is_self'] else 'blue' for p in top_pairs],
                line=dict(width=1, color='white')
            ),
            text=[f"F{p['feature_1']}{'(self)' if p['is_self'] else '↔F' + str(p['feature_2'])}"
                  for p in top_pairs],
            hovertemplate="<b>%{text}</b><br>Count: %{x}<br>Abs Shapley: %{y:.6f}<extra></extra>",
            name='Interactions'
        ),
        row=3, col=1
    )

    # 6. Feature count distribution
    active_counts = feature_counts[feature_counts > 0]

    fig.add_trace(
        go.Histogram(
            x=active_counts,
            nbinsx=30,
            name='Feature Activity',
            marker_color='green',
            hovertemplate="Count range: %{x}<br>Features: %{y}<extra></extra>"
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Shapley Feature Interactions Dashboard<br>{model_name} | {layer_name} | {total_samples} samples | Top {top_k} pairs",
        height=1800,
        showlegend=False,
        font=dict(size=10)
    )

    # Update axes
    fig.update_xaxes(title_text="Absolute Shapley-Taylor Index", row=1, col=1)
    # CCDF axes already set above with log scale
    fig.update_xaxes(title_text="Abs Shapley Value", row=2, col=2)
    fig.update_xaxes(title_text="Occurrence Count", row=3, col=1)
    fig.update_yaxes(title_text="Absolute Shapley-Taylor Index", row=3, col=1)
    fig.update_xaxes(title_text="Feature Occurrence Count", row=3, col=2)
    fig.update_yaxes(title_text="Number of Features", row=3, col=2)

    # Calculate summary statistics
    num_active_features = len(results['active_features'])
    num_self_interactions = len([p for p in top_pairs if p['is_self']])
    # When using absolute values, we can still track the sign from the signed sum
    num_positive = len([p for p in top_pairs if p['signed_sum'] > 0])
    num_negative = len([p for p in top_pairs if p['signed_sum'] < 0])

    # Generate HTML with embedded data
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shapley Feature Interactions Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1600px;
                margin: 0 auto;
            }}
            .header {{
                background: rgba(255, 255, 255, 0.95);
                color: #2c3e50;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 32px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }}
            .stat-card {{
                background: rgba(255, 255, 255, 0.95);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }}
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            .stat-value {{
                font-size: 28px;
                font-weight: bold;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stat-label {{
                color: #7f8c8d;
                margin-top: 5px;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .chart-container {{
                background: rgba(255, 255, 255, 0.95);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-bottom: 25px;
            }}
            .top-pairs {{
                background: rgba(255, 255, 255, 0.95);
                padding: 25px;
                border-radius: 15px;
                margin-top: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .pair-item {{
                padding: 15px;
                border-left: 4px solid transparent;
                margin-bottom: 10px;
                background: #f8f9fa;
                border-radius: 8px;
                transition: all 0.3s ease;
            }}
            .pair-item:hover {{
                background: #e9ecef;
                border-left-color: #667eea;
                transform: translateX(5px);
            }}
            .self-interaction {{
                background: #ffe6e6;
                border-left-color: #e74c3c;
            }}
            .positive-interaction {{
                border-left-color: #3498db;
            }}
            .negative-interaction {{
                border-left-color: #e67e22;
            }}
            .pair-rank {{
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-weight: bold;
                margin-right: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 Shapley Feature Interactions Dashboard</h1>
                <p><strong>Model:</strong> {model_name} | <strong>Layer:</strong> {layer_name}</p>
                <p>Analysis of feature interactions using Shapley-Taylor decomposition</p>
            </div>

            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{num_active_features:,}</div>
                    <div class="stat-label">Active Features</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['total_pairs']:,}</div>
                    <div class="stat-label">Feature Pairs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_samples}</div>
                    <div class="stat-label">Samples Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{num_self_interactions}</div>
                    <div class="stat-label">Self-Interactions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{num_positive}</div>
                    <div class="stat-label">Positive Interactions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{num_negative}</div>
                    <div class="stat-label">Negative Interactions</div>
                </div>
            </div>

            <div class="chart-container">
                <div id="plotly-chart"></div>
            </div>

            <div class="top-pairs">
                <h2>Top 15 Feature Interactions (Shapley-Taylor Indices)</h2>
                {"".join([f'''
                <div class="pair-item {'self-interaction' if p['is_self'] else 'positive-interaction' if p['signed_sum'] > 0 else 'negative-interaction'}">
                    <span class="pair-rank">#{i+1}</span>
                    <strong>Feature {p['feature_1']} {'(self)' if p['is_self'] else '↔ Feature ' + str(p['feature_2'])}</strong><br>
                    <small><strong>Abs Shapley Avg:</strong> {p['average_abs']:.6f} | <strong>Total Abs:</strong> {p['abs_sum']:.6f} | <strong>Count:</strong> {p['count']}</small><br>
                    <small style="color: #7f8c8d;">
                        {feature_descriptions.get(p['feature_1'], f'Feature {p["feature_1"]}')}
                        {'' if p['is_self'] else '↔ ' + feature_descriptions.get(p['feature_2'], f'Feature {p["feature_2"]}')}
                    </small>
                </div>
                ''' for i, p in enumerate(top_pairs[:15])])}
            </div>
        </div>

        <script>
            var plotlyData = {fig.to_json()};
            Plotly.newPlot('plotly-chart', plotlyData.data, plotlyData.layout, {{responsive: true}});
        </script>
    </body>
    </html>
    """

    # Save dashboard
    if output_path is None:
        output_path = Path(__file__).parent / "results" / "viz" / f"shapley_dashboard_{model_name}_{layer_name.replace('.', '_')}_{total_samples}samples.html"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✅ Dashboard saved to: {output_path}")
    return str(output_path)


def load_and_visualize_shapley_results(
    npz_path: str,
    model_name: str = "TinyStories-33M",
    crosscoder_name: str = "ckubmeg1",
    top_k: int = 50,
    output_path: Optional[str] = None,
    fetch_neuronpedia: bool = False
) -> str:
    """
    Load Shapley results from NPZ file and create visualization dashboard.

    Args:
        npz_path: Path to the NPZ file with Shapley results
        model_name: Name of the model for display
        crosscoder_name: Name of the crosscoder (for loading explanations)
        top_k: Number of top pairs to visualize
        output_path: Where to save the dashboard
        fetch_neuronpedia: Whether to fetch feature descriptions (deprecated, uses CSV)

    Returns:
        Path to saved dashboard
    """
    # Load the NPZ file
    data = np.load(npz_path)

    # Extract arrays
    pairwise_signed_sum = data['pairwise_signed_sum']
    pairwise_counts = data['pairwise_counts']
    feature_counts = data['feature_counts']
    single_indices_sum = data['single_indices_sum']
    pairwise_abs_sum = data.get('pairwise_abs_sum', None)

    # Extract metadata
    total_samples = int(data.get('total_samples', 0))
    start_hook = str(data.get('start_hook', 'unknown'))
    end_hook = str(data.get('end_hook', 'unknown'))

    # Try to extract crosscoder name from metadata if available
    stored_crosscoder = data.get('crosscoder_name', None)
    if stored_crosscoder:
        crosscoder_name = str(stored_crosscoder)

    # Create layer name from hooks
    layer_name = f"{start_hook} → {end_hook}"

    return create_shapley_dashboard(
        pairwise_signed_sum=pairwise_signed_sum,
        pairwise_counts=pairwise_counts,
        feature_counts=feature_counts,
        single_indices_sum=single_indices_sum,
        pairwise_abs_sum=pairwise_abs_sum,
        total_samples=total_samples,
        model_name=model_name,
        layer_name=layer_name,
        crosscoder_name=crosscoder_name,
        top_k=top_k,
        output_path=output_path,
        fetch_neuronpedia=fetch_neuronpedia
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create Shapley interaction dashboard')
    parser.add_argument('--npz', type=str, help='Path to NPZ file with Shapley results')
    parser.add_argument('--model', type=str, default='TinyStories-33M', help='Model name')
    parser.add_argument('--crosscoder', type=str, default='ckubmeg1', help='Crosscoder name for loading explanations')
    parser.add_argument('--top-k', type=int, default=50, help='Number of top pairs to show')
    parser.add_argument('--output', type=str, help='Output path for dashboard')
    parser.add_argument('--fetch-descriptions', action='store_true', help='Deprecated - now loads from CSV')

    args = parser.parse_args()

    if args.npz:
        dashboard_path = load_and_visualize_shapley_results(
            npz_path=args.npz,
            model_name=args.model,
            crosscoder_name=args.crosscoder,
            top_k=args.top_k,
            output_path=args.output,
            fetch_neuronpedia=False  # Always use CSV now
        )
        print(f"Dashboard created: {dashboard_path}")
    else:
        print("Please provide an NPZ file path with --npz")