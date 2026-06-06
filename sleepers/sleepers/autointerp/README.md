# Autointerpretation Pipeline

This directory contains a suite of Python scripts and Jupyter notebooks for automatically generating and evaluating explanations for features learned by a neural network, likely in the context of language models and "CrossCoders".

## Overview

The pipeline consists of the following main steps:

1.  **Collect Feature Activations**: Extract feature activations from a model over a dataset.
2.  **Generate Explanations**: Use a Large Language Model (LLM) via Azure OpenAI to generate human-readable explanations for each feature based on its activations.
3.  **Evaluate Explanations**: Use an LLM to evaluate the quality and accuracy of the generated explanations.
4.  **Analyze Metrics**: Compute specific metrics related to CrossCoders (e.g., specificity, sensitivity) and visualize them.
5.  **Interactive Dashboard**: Explore features, their activating examples, explanations, and evaluation metrics.

## File Structure and Key Components

*   `1_collect_autointerp_data.py`: Script to collect feature activations from your model and dataset. Saves data to the `collected_activation_data/` directory.
*   `2_get_explanations.py`: Jupyter notebook to load collected activations, format them, call Azure OpenAI to generate explanations, and save them to a CSV in `autointerp_data/`.
*   `3_autointerp_eval.py`: Script to load generated explanations and use Azure OpenAI to evaluate their accuracy, saving evaluation metrics to a CSV in `autointerp_data/`.
*   `4_autointerp_CC_metrics.py`: Script to calculate and plot CrossCoder-specific metrics (specificity, sensitivity) based on the explanations.
*   `5_autointerp_dashboard.py`: A script to create an IPython widget/Dash app to explore the results.

*   `utils/`: Directory containing utility modules:
    *   `__init__.py`: Makes `utils` a Python package.
    *   `llm_autointerp.py`: Handles communication with the Azure OpenAI API (GPT-4o) for generating and evaluating explanations. Requires an API key.
    *   `autointerp_prompts.py`: Contains the system prompts (`SYSTEM_EXPLAINER`, `SYSTEM_EVALUATOR`) used for interacting with the LLM.
    *   `data_formatting_util.py`: Provides utilities for formatting feature activation data (e.g., extracting top/bottom activating examples, context strings) to create prompts for the LLM.
    *   `activation_util.py` (or `autointerp_util.py`): Contains general utility functions, including `get_activations_batch` for processing model inputs and extracting activations.

*   `collected_activation_data/`: Directory where raw feature activation data is stored.
*   `autointerp_data/`: Directory for storing explanation data and evaluation metrics generated from API calls.
*   `.env` (You need to create this in the root directory): File containig `AZURE_OPENAI_API_KEY`.


## Running the Pipeline

Make sure to update import paths in the scripts if you've moved files (e.g., `from utils.llm_autointerp import ...`).

1.  **Collect Feature Activations**:
    *   Modify `1_collect_autointerp_data.py` (model, CrossCoder, dataset).
    *   Run: `python 1_collect_autointerp_data.py`
    *   Populates `collected_activation_data/`.

2.  **Generate Explanations**:
    *   Ensure `2_get_explanations.py` loads from `collected_activation_data/`.
    *   Run the notebook or script: `python 2_get_explanations.py` (if converted to script) or run cells in Jupyter.
    *   Saves explanations CSV to `autointerp_data/`.

3.  **Evaluate Explanations**:
    *   Modify `3_autointerp_eval.py` to load explanations from correct location.
    *   Run: `python 3_autointerp_eval.py`
    *   Saves evaluation metrics CSV to `autointerp_data/`.

4.  **Analyze CrossCoder Metrics**:
    *   Modify `4_autointerp_CC_metrics.py` to load relevant data.
    *   Run: `python 4_autointerp_CC_metrics.py`
    *   Produces plots/results.

5.  **View Dashboard**:
    *   Ensure paths in `5_autointerp_dashboard.py` are correct.
    *   Run: `python 5_autointerp_dashboard.py`
    *   Access via browser if it's a web app, or view widget in IPython.


