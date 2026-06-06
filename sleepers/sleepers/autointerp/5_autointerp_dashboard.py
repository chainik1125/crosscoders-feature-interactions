## Creates a local dashboard in an python notebook (using ipywidgets)
# This loads activations, explanations and metrics
# It displays:
#  - top 5 interacting tokens for each feature
#  - sensitivity and specificity of each feature
#  - explanation of each feature
#  - the sequence containing the top activating token, with every token highlighted according to its activation strength

# %%
# CELL 1: Data Loading and Preparation
import pandas as pd
import pickle
import os

print("--- Cell 1: Data Loading ---")

# Configuration
crosscoder_name = "v7128kc4"
ACTIVATIONS_PKL_FILENAME = f"CC-{crosscoder_name}_10000-samples.pkl"
ACTIVATIONS_PKL_PATH = os.path.join("autointerp_data/collected_activation_data", ACTIVATIONS_PKL_FILENAME)
EXPLANATIONS_CSV_PATH = f'autointerp_data/explanations_{crosscoder_name}_20250510_174018.csv'
METRICS_CSV_PATH = f'autointerp_data/autointerp_eval_metrics_{crosscoder_name}.csv'

# Initialize data variables
all_sequences = []
raw_activations_dict = {} 
explanations_df = pd.DataFrame()
metrics_df = pd.DataFrame()
data_load_success = True


with open(ACTIVATIONS_PKL_PATH, 'rb') as f:
    loaded_act_data = pickle.load(f)
all_sequences = loaded_act_data.get('sequences', [])
raw_activations_dict = loaded_act_data.get('activations', {})
print(f"Successfully loaded activations pickle file from {ACTIVATIONS_PKL_PATH}")
if not all_sequences or not raw_activations_dict:
    print("Warning: Activations data from pickle might be empty or missing 'sequences'/'activations' keys.")


explanations_df = pd.read_csv(EXPLANATIONS_CSV_PATH)
print(f"Successfully loaded {EXPLANATIONS_CSV_PATH}")

metrics_df = pd.read_csv(METRICS_CSV_PATH)
print(f"Successfully loaded {METRICS_CSV_PATH}")


# %%
# CELL 2: Dashboard Implementation
import ipywidgets as widgets
from IPython.display import display, HTML
import html # For html.escape
import numpy as np # Added for cases where it might be used by imported functions or future enhancements
print(metrics_df.head())
print(metrics_df.columns)

print(explanations_df.head())
print(explanations_df.columns)

print("--- Cell 2: Dashboard ---")

# Attempt to import from the local data_formatting_util.py file
try:
    from sleepers.autointerp.util.data_formatting_util import format_activations
    # _format_context_string is not directly used in this cell's display logic after changes,
    # but format_activations might use it internally.
except ImportError:
    print("Error: Could not import 'format_activations' from 'sleepers.autointerp.util.data_formatting_util'.")
    print("Ensure 'data_formatting_util.py' is in the specified path and there are no circular imports.")
    # Provide a dummy function if import fails, so the dashboard can attempt to load
    def format_activations(*args, **kwargs):
        print("Warning: 'format_activations' is not available due to import error.")
        return None 

# Check if data loaded successfully from Cell 1
if 'data_load_success' in globals() and not data_load_success:
    print("Dashboard cannot be initialized properly due to data loading errors in the previous cell.")
    # Optionally display an HTML error in a Jupyter environment if this cell were to run independently
    # display(HTML("<p style='color:red;'>Critical data loading errors occurred. Dashboard functionality will be limited.</p>"))
elif not all_sequences or not raw_activations_dict : # Check core data needed for features
    print("Core activation data (all_sequences or raw_activations_dict) not loaded. Dashboard cannot fully initialize.")
else:
    # --- Dashboard Display Functions ---
    def get_color_for_activation(activation_value, max_abs_val):
        """Gets a color based on activation value."""
        if max_abs_val == 0: return 'rgba(0,0,0,0.05)' # Lighter default for zero/no significant activation
        intensity = min(abs(activation_value) / max_abs_val, 1.0) * 0.8 # Max alpha 0.8
        if activation_value > 0:
            return f'rgba(255, 0, 0, {intensity})'  # Red for positive
        elif activation_value < 0:
            return f'rgba(0, 0, 255, {intensity})'  # Blue for negative
        return 'rgba(0,0,0,0.05)' # Very light for zero

    output_area = widgets.Output() # Define output_area in a scope accessible by display_feature_data

    def display_feature_data(feature_id):
        output_area.clear_output(wait=True)
        
        with output_area:
            display(HTML(f"<h3>Feature Details for ID: {feature_id}</h3>"))

            # 1. Display Explanation
            if not explanations_df.empty and 'feature_id' in explanations_df.columns and 'explanation' in explanations_df.columns:
                explanation_series = explanations_df[explanations_df['feature_id'] == feature_id]['explanation']
                if not explanation_series.empty:
                    display(HTML(f"<b>Explanation:</b> {html.escape(str(explanation_series.iloc[0]))}"))
                else:
                    display(HTML(f"No explanation found for feature ID {feature_id}."))
            else:
                display(HTML("Explanations data not available or not in expected format. Check `explanations_df.columns` and file content."))

            # 2. Display Metrics
            # Check for the columns needed for calculation, not pre-calculated sensitivity/specificity
            required_metric_cols = ['feature_id', 'true_positives', 'false_negatives', 'true_negatives', 'false_positives']
            if not metrics_df.empty and all(col in metrics_df.columns for col in required_metric_cols):
                metric_row = metrics_df[metrics_df['feature_id'] == feature_id]
                if not metric_row.empty:
                    tp = metric_row['true_positives'].iloc[0]
                    fn = metric_row['false_negatives'].iloc[0]
                    tn = metric_row['true_negatives'].iloc[0]
                    fp = metric_row['false_positives'].iloc[0]

                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    display(HTML(f"<b>Sensitivity:</b> {sensitivity:.2f}  |  <b>Specificity:</b> {specificity:.2f}"))
                else:
                    display(HTML(f"No metrics found for feature ID {feature_id}."))
            else:
                display(HTML("Metrics data not available or not in expected format. Ensure `metrics_df` has columns: " + ", ".join(required_metric_cols)))
            
            display(HTML("<hr>"))

            # 3. Display Top Activating Tokens
            if not all_sequences or not raw_activations_dict or feature_id not in raw_activations_dict:
                display(HTML("Activation data (raw_activations_dict) not available for this feature."))
                return

            if 'format_activations' not in globals() or not callable(format_activations):
                display(HTML("<p style='color:red;'>Critical error: 'format_activations' function is not available. Cannot display token info.</p>"))
                return

            formatted_acts = format_activations(
                feature_id, 
                all_sequences, 
                raw_activations_dict, 
                n_top=5, 
                n_bottom=0, # Not requesting bottom activations
                context_window=5, 
                format_for_eval=True 
            )

            if formatted_acts:
                display(HTML("<h4>Top 5 Activating Token Contexts:</h4>"))
                top_acts = formatted_acts.get('top_activations', [])
                if top_acts:
                    for ex in top_acts:
                        act_val_str = f"{ex['activation']:.2f}" if isinstance(ex.get('activation'), (int, float)) else "N/A"
                        
                        context_html = ""
                        raw_context = str(ex.get('context', ''))
                        parts = raw_context.split("<<")
                        if len(parts) > 1:
                            context_html += html.escape(parts[0])
                            sub_parts = parts[1].split(">>")
                            if len(sub_parts) > 1:
                                context_html += f"<span style='color: red; font-weight: bold;'>{html.escape(sub_parts[0])}</span>"
                                context_html += html.escape(sub_parts[1])
                            else: # No closing >>, treat as literal
                                context_html += "&lt;&lt;" + html.escape(parts[1])
                        else: # No << found, treat as literal
                            context_html = html.escape(raw_context)
                        
                        display(HTML(f"<b>Activation: {act_val_str}</b> - Context: {context_html}"))
                else:
                    display(HTML("No top activations found or returned by format_activations."))
                
                display(HTML("<hr>"))

                # 4. Display Sequence with Highlighted Activations for the top activating token
                if top_acts:
                    # Ensure top_acts[-1] is valid and has 'sequence_id'
                    top_example_for_highlight = top_acts[-1] if top_acts else None
                    
                    if top_example_for_highlight and 'sequence_id' in top_example_for_highlight:
                        seq_id = top_example_for_highlight['sequence_id']

                        if isinstance(seq_id, (int, np.integer)) and seq_id >= 0 and seq_id < len(all_sequences):
                            sequence_tokens = all_sequences[seq_id]
                            # Ensure feature_id exists in raw_activations_dict and seq_id in its sub-dictionary
                            if feature_id in raw_activations_dict and seq_id in raw_activations_dict[feature_id]:
                                activation_values_for_seq = raw_activations_dict[feature_id][seq_id]
                            else:
                                activation_values_for_seq = [] # Default to empty if not found

                            if activation_values_for_seq and len(activation_values_for_seq) == len(sequence_tokens):
                                display(HTML("<h4>Top Activating Sequence (Highlighted by Activation Strength):</h4>"))
                                max_abs_val_in_seq = max(abs(val) for val in activation_values_for_seq) if activation_values_for_seq else 0
                                if max_abs_val_in_seq == 0: max_abs_val_in_seq = 1 

                                html_parts = []
                                for i, token_text in enumerate(sequence_tokens):
                                    act_val = activation_values_for_seq[i]
                                    color = get_color_for_activation(act_val, max_abs_val_in_seq)
                                    escaped_token_text = html.escape(str(token_text))
                                    html_parts.append(f"<span style='background-color:{color}; padding: 1px; margin: 0.5px; border-radius: 3px;'>{escaped_token_text}</span>")
                                
                                full_html_sequence = "".join(html_parts)
                                display(HTML(f"<div style='font-family: monospace; white-space: pre-wrap; line-height: 1.8;'>{full_html_sequence}</div>"))
                            elif not activation_values_for_seq: # Handles empty list
                                display(HTML("Could not retrieve activation values for the top sequence."))
                            else: # Mismatched lengths
                                display(HTML(f"<p style='color:orange;'>Warning: Mismatch between token count ({len(sequence_tokens)}) and activation values ({len(activation_values_for_seq)}) for sequence ID {seq_id}.</p>"))

                        else:
                            display(HTML(f"<p style='color:red;'>Error: Sequence ID ({seq_id}) for top activation is invalid or out of bounds ({len(all_sequences)} sequences available).</p>"))
                    else: # No top_example_for_highlight or missing sequence_id
                         display(HTML("<p style='color:orange;'>Could not identify a valid top example for highlighting.</p>"))
                else: # No top_acts
                    display(HTML("No top activations to select a sequence for highlighting."))
            else: # No formatted_acts
                display(HTML("<p style='color:orange;'>Could not format activations using 'format_activations'.</p>"))

    # --- Initialize Widgets ---
    # Determine features with non-zero activations
    epsilon = 1e-9 # Small value to consider an activation non-zero
    features_with_non_zero_activations = set()
    if raw_activations_dict:
        for feat_id, sequences_data in raw_activations_dict.items():
            for activation_list in sequences_data.values():
                if any(abs(act_val) > epsilon for act_val in activation_list):
                    features_with_non_zero_activations.add(feat_id)
                    break # Found non-zero for this feature, move to next feat_id
    
    # Populate dropdown ONLY with features that have non-zero activations
    # and are numeric (convertible to int)
    feature_ids_options = []
    if features_with_non_zero_activations:
        processed_ids_for_dropdown = set() # To ensure uniqueness after int conversion
        # Sort first by the original feature ID type if they are mixed, then convert to int
        # This handles cases where feature IDs might be strings like '0', '1' etc.
        sorted_raw_feat_ids = []
        try:
            # Attempt to sort assuming they might be numeric or string representations of numbers
            sorted_raw_feat_ids = sorted(list(features_with_non_zero_activations), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x if isinstance(x, int) else float('inf'))
        except TypeError: # Fallback if mixed types cause sorting issues directly
            sorted_raw_feat_ids = sorted(list(str(x) for x in features_with_non_zero_activations))


        for fid in sorted_raw_feat_ids:
            try:
                int_fid = int(fid) # Ensure it's a base-10 integer
                if int_fid not in processed_ids_for_dropdown:
                    feature_ids_options.append(int_fid)
                    processed_ids_for_dropdown.add(int_fid)
            except (ValueError, TypeError):
                print(f"Warning: Feature ID {fid} from activations is non-numeric or problematic and will be skipped for dropdown.")
        
        feature_ids_options.sort() # Final sort of integer IDs

    if not feature_ids_options:
        print("Critical Error: No valid feature IDs with non-zero activations found. Dashboard cannot be initialized.")
        display(HTML("<p style='color:red;'>Critical Error: No features with non-zero activations available. Dashboard cannot operate.</p>"))
    else:
        feature_dropdown = widgets.Dropdown(
            options=feature_ids_options,
            description='Feature ID:',
            value=feature_ids_options[0],
            disabled=False
        )

        def on_dropdown_change(change):
            if change.new is not None: 
                display_feature_data(change.new)

        feature_dropdown.observe(on_dropdown_change, names='value')

        # Initial display
        print("Dashboard initialized in Cell 2. Select a feature ID to view details.")
        display(widgets.VBox([feature_dropdown, output_area]))
        if feature_dropdown.value is not None:
            display_feature_data(feature_dropdown.value)
        else:
            print("No default feature ID selected for initial display (should not happen if options exist).")



# %%
