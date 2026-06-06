import numpy as np
# %%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio

# base_dir = '/home/anna/Documents/MARS_crosscoders/crosscoders-feature-interactions/sleepers/sleepers/autointerp'
base_dir = '/root/crosscoders-feature-interactions/sleepers/sleepers/autointerp/autointerp_data'
# Load the metrics for two different crosscoders
crosscoder_name1 = "daifvx03"
crosscoder_name2 = "ckubmeg1"  # Second crosscoder to compare
metrics_df1 = pd.read_csv(f'{base_dir}/autointerp_eval_metrics_{crosscoder_name1}_withhate.csv')
metrics_df2 = pd.read_csv(f'{base_dir}/autointerp_eval_metrics_{crosscoder_name2}.csv')

# %%
# Calculate sensitivity and specificity for both crosscoders

def calculate_metrics(df):
    for index, row in df.iterrows():
        if row['true_positives']==0:
            continue
        else:
            sensitivity = row['true_positives'] / (row['true_positives'] + row['false_negatives'])
        if row['true_negatives']==0:
            pass
        else:
            specificity = row['true_negatives'] / (row['true_negatives'] + row['false_positives'])
        df.loc[index, 'sensitivity'] = sensitivity
        df.loc[index, 'specificity'] = specificity
    return df

metrics_df1 = calculate_metrics(metrics_df1)
metrics_df2 = calculate_metrics(metrics_df2)

# print the explanations of the bottom 5 features by sensitivity for first crosscoder
metrics_df1_sorted = metrics_df1.sort_values(by='sensitivity', ascending=True)
for index, row in metrics_df1_sorted.head(5).iterrows():
    print(f"Feature ID: {row['feature_id']}")
    print(f"Explanation: {row['explanation']}")
    print(f"Sensitivity: {row['sensitivity']}")
    print(f"Specificity: {row['specificity']}")
    print('--------------------------------')

# %%
# Create subplots to compare sensitivity and specificity distributions
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # Plot sensitivity distributions
# axes[0, 0].hist(metrics_df1['sensitivity'], bins=20, edgecolor='black', color='skyblue', alpha=0.7)
# axes[0, 0].set_title(f'Sensitivity Distribution - {crosscoder_name1}', fontsize=12, fontweight='bold')
# axes[0, 0].set_xlabel('Sensitivity', fontsize=10)
# axes[0, 0].set_ylabel('Frequency', fontsize=10)
# axes[0, 0].grid(axis='y', alpha=0.3)
# axes[0, 0].axvline(metrics_df1['sensitivity'].mean(), color='red', linestyle='dashed', linewidth=2, 
#                   label=f'Mean: {metrics_df1["sensitivity"].mean():.2f}')
# axes[0, 0].legend()

# axes[0, 1].hist(metrics_df2['sensitivity'], bins=20, edgecolor='black', color='skyblue', alpha=0.7)
# axes[0, 1].set_title(f'Sensitivity Distribution - {crosscoder_name2}', fontsize=12, fontweight='bold')
# axes[0, 1].set_xlabel('Sensitivity', fontsize=10)
# axes[0, 1].set_ylabel('Frequency', fontsize=10)
# axes[0, 1].grid(axis='y', alpha=0.3)
# axes[0, 1].axvline(metrics_df2['sensitivity'].mean(), color='red', linestyle='dashed', linewidth=2, 
#                   label=f'Mean: {metrics_df2["sensitivity"].mean():.2f}')
# axes[0, 1].legend()

# # Plot specificity distributions
# axes[1, 0].hist(metrics_df1['specificity'], bins=20, edgecolor='black', color='green', alpha=0.7)
# axes[1, 0].set_title(f'Specificity Distribution - {crosscoder_name1}', fontsize=12, fontweight='bold')
# axes[1, 0].set_xlabel('Specificity', fontsize=10)
# axes[1, 0].set_ylabel('Frequency', fontsize=10)
# axes[1, 0].grid(axis='y', alpha=0.3)
# axes[1, 0].axvline(metrics_df1['specificity'].mean(), color='red', linestyle='dashed', linewidth=2, 
#                   label=f'Mean: {metrics_df1["specificity"].mean():.2f}')
# axes[1, 0].legend()

# axes[1, 1].hist(metrics_df2['specificity'], bins=20, edgecolor='black', color='green', alpha=0.7)
# axes[1, 1].set_title(f'Specificity Distribution - {crosscoder_name2}', fontsize=12, fontweight='bold')
# axes[1, 1].set_xlabel('Specificity', fontsize=10)
# axes[1, 1].set_ylabel('Frequency', fontsize=10)
# axes[1, 1].grid(axis='y', alpha=0.3)
# axes[1, 1].axvline(metrics_df2['specificity'].mean(), color='red', linestyle='dashed', linewidth=2, 
#                   label=f'Mean: {metrics_df2["specificity"].mean():.2f}')
# axes[1, 1].legend()

# plt.tight_layout()
# plt.savefig(f'{base_dir}/crosscoder_comparison.pdf')
# plt.show()


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------------
# 1. 2×2 canvas with the same subplot titles you had in Matplotlib
# ------------------------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f"Sensitivity Distribution - Unpenalized",# {crosscoder_name1}",
        f"Sensitivity Distribution - Penalized",# {crosscoder_name2}",
        f"Specificity Distribution - Unpenalized",# {crosscoder_name1}",
        f"Specificity Distribution - Penalized"# {crosscoder_name2}",
    ]
)

# ------------------------------------------------------------------------------
# 2. helper that adds histogram, dashed-red mean line, and mean label
# ------------------------------------------------------------------------------
def add_hist(df, column, colour, row, col, series_label):
    vals = df[column].values
    mu   = vals.mean()

    # --- histogram trace -------------------------------------------------------
    fig.add_trace(
        go.Histogram(
            x            = vals[~np.isnan(vals)],
            nbinsx       = 20,
            marker_color = colour,
            opacity      = 0.7,
            name         = series_label,          # shows in that subplot’s legend
            showlegend   = True,
        ),
        row=row, col=col
    )

    # --- height of tallest bin (for placing line + label) ----------------------
    counts, _ = np.histogram(vals[~np.isnan(vals)], bins=20)
    y_max     = counts.max()

    # --- dashed red mean line --------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=[mu, mu], y=[0, y_max],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name=f"Mean ({mu:.2f})",              # legend entry for the line
            showlegend=False                     # toggle to True if you prefer
        ),
        row=row, col=col
    )

    # --- annotation just above the line ---------------------------------------
    fig.add_annotation(
        x=mu, y=y_max * 1.05,                    # 5 % above tallest bar
        text=f"Mean: {mu:.2f}",
        showarrow=False,
        font=dict(color='red', size=20),
        xanchor='center', yanchor='bottom',
        row=row, col=col
    )

# sensitivities (row 1)
add_hist(metrics_df1, 'sensitivity', 'skyblue', row=1, col=1,
         series_label=f"Sensitivity – unpenalized") # {crosscoder_name1}")
add_hist(metrics_df2, 'sensitivity', 'skyblue', row=1, col=2,
         series_label=f"Sensitivity – penalized") # {crosscoder_name2}")

# specificities (row 2)
add_hist(metrics_df1, 'specificity', 'green',   row=2, col=1,
         series_label=f"Specificity – unpenalized") # {crosscoder_name1}")
add_hist(metrics_df2, 'specificity', 'green',   row=2, col=2,
         series_label=f"Specificity – penalized") # {crosscoder_name2}")

# ------------------------------------------------------------------------------
# 3. cosmetic tweaks to mimic the original look
# ------------------------------------------------------------------------------
for r in (1, 2):
    for c in (1, 2):
        fig.update_xaxes(
            title_text="Sensitivity" if r == 1 else "Specificity",
            row=r, col=c)
        fig.update_yaxes(
            title_text="Frequency",
            showgrid=True, gridwidth=0.3, gridcolor='rgba(0,0,0,0.3)',
            row=r, col=c)

fig.update_layout(
    bargap   = 0.05,
    width    = 900,
    height   = 700,
    legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
    margin=dict(t=80),
)

# ------------------------------------------------------------------------------
# 4. display / save
# ------------------------------------------------------------------------------
#fig.show()                                                   # interactive
# Turn off MathJax for plotly.io to avoid rendering issues

#pio.kaleido.scope.mathjax = None
fig.update_xaxes(tickfont=dict(size=20),title_font=dict(size=20))
fig.update_yaxes(tickfont=dict(size=20),title_font=dict(size=20))
fig.update_layout(legend=dict(font=dict(size=16)))
# Adjust the legend to make it fit better
fig.update_layout(
    legend=dict(
        orientation='h',          # Horizontal orientation
        yanchor='bottom',         # Anchor to bottom
        y=-0.25,                  # Move it further down to avoid overlap
        xanchor='center',         
        x=0.5,
        font=dict(size=14),       # Slightly larger font for better readability
        itemsizing='constant',    # Make legend items consistent size
        itemwidth=30,             # Control width of legend items
        tracegroupgap=5           # Reduce gap between legend groups
    )
)

# Alternative: if the legend is still too crowded, we can simplify the labels
for i in range(len(fig.data)):
    if 'unpenalized' in fig.data[i].name:
        fig.data[i].name = fig.data[i].name.replace('unpenalized', 'λ=0')
    elif 'penalized' in fig.data[i].name:
        fig.data[i].name = fig.data[i].name.replace('penalized', 'λ=1000')


fig.write_html(f"{base_dir}/crosscoder_comparison.html")     # identical PDF (needs kaleido)


# %%
# print mean and std of sensitivity and specificity for both crosscoders
print(f"Average sensitivity for {crosscoder_name1}: {metrics_df1['sensitivity'].mean():.2f} (std: {metrics_df1['sensitivity'].std():.2f})")
print(f"Average specificity for {crosscoder_name1}: {metrics_df1['specificity'].mean():.2f} (std: {metrics_df1['specificity'].std():.2f})")
print(f"Average sensitivity for {crosscoder_name2}: {metrics_df2['sensitivity'].mean():.2f} (std: {metrics_df2['sensitivity'].std():.2f})")
print(f"Average specificity for {crosscoder_name2}: {metrics_df2['specificity'].mean():.2f} (std: {metrics_df2['specificity'].std():.2f})")

# %%
# print the percentage of features with sensitivity > 0.5 for both crosscoders
print(f"{crosscoder_name1}: {metrics_df1[metrics_df1['sensitivity'] > 0.5].shape[0] / metrics_df1.shape[0] * 100:.2f}% features with sensitivity > 0.5")
print(f"{crosscoder_name2}: {metrics_df2[metrics_df2['sensitivity'] > 0.5].shape[0] / metrics_df2.shape[0] * 100:.2f}% features with sensitivity > 0.5")
print('--------------------------------')
# print the percentage of features with specificity > 0.5 for both crosscoders
print(f"{crosscoder_name1}: {metrics_df1[metrics_df1['specificity'] > 0.5].shape[0] / metrics_df1.shape[0] * 100:.2f}% features with specificity > 0.5")
print(f"{crosscoder_name2}: {metrics_df2[metrics_df2['specificity'] > 0.5].shape[0] / metrics_df2.shape[0] * 100:.2f}% features with specificity > 0.5")
print('--------------------------------')
# same for 0.9
print(f"{crosscoder_name1}: {metrics_df1[metrics_df1['specificity'] > 0.9].shape[0] / metrics_df1.shape[0] * 100:.2f}% features with specificity > 0.9")
print(f"{crosscoder_name2}: {metrics_df2[metrics_df2['specificity'] > 0.9].shape[0] / metrics_df2.shape[0] * 100:.2f}% features with specificity > 0.9")
print('--------------------------------')
# print the percentage of features with sensitivity > 0.9 for both crosscoders
print(f"{crosscoder_name1}: {metrics_df1[metrics_df1['sensitivity'] > 0.9].shape[0] / metrics_df1.shape[0] * 100:.2f}% features with sensitivity > 0.9")
print(f"{crosscoder_name2}: {metrics_df2[metrics_df2['sensitivity'] > 0.9].shape[0] / metrics_df2.shape[0] * 100:.2f}% features with sensitivity > 0.9")
print('--------------------------------')
# %%










