#!/usr/bin/env python3
"""Quick test with very conservative parameters to ensure it works."""

import yaml

# Create a very conservative config
conservative_config = {
    'dataset': {
        'name': "mars-jason-25/tiny_stories_instruct_sleeper_data",
        'split': "train",
        'filter_training': True
    },
    'model': {
        'base_model_repo': "roneneldan/TinyStories-Instruct-33M",
        'lora_model_repo': "mars-jason-25/tiny-stories-33M-TSdata-ft1",
        'cache_dir': None,
        'dtype': None
    },
    'crosscoder': {
        'wandb_entity': "dmitry2-uiuc",
        'wandb_project': "sleeper-model-diffing",
        'wandb_run_id': "86u64trx",
        'artifacts_dir': "../../.wandb_artifacts"
    },
    'comparison': {
        'num_stories': 1,    # Just 1 story for speed
        'layer': 0
    },
    'shapley_taylor': {
        'max_features_per_neuron': 5,  # Very few features
        'num_samples': 50,             # Very few samples
        'threshold': 0.1,              # Very high threshold
        'small_threshold': 0.00000001,
        'verbose': True
    },
    'existing_method': {},
    'analysis': {
        'plot_sample_size': 1000,
        'correlation_threshold': 10,
        'save_results': False,
        'output_dir': "comparison_results"
    },
    'plotting': {
        'figure_size': [10, 8],
        'dpi': 300,
        'show_plot': True,
        'save_plot': True,
        'plot_filename': "quick_test_comparison.png",
        'both_methods_color': "blue",
        'both_methods_alpha': 0.6,
        'both_methods_size': 20,
        'shapley_only_color': "red",
        'shapley_only_marker': "^",
        'shapley_only_alpha': 0.4,
        'shapley_only_size': 15,
        'existing_only_color': "green",
        'existing_only_marker': "s",
        'existing_only_alpha': 0.4,
        'existing_only_size': 15
    },
    'performance': {
        'device': "auto",
        'timeout_minutes': 10,
        'memory_cleanup': True
    }
}

# Save conservative config
with open('conservative_test.yaml', 'w') as f:
    yaml.dump(conservative_config, f, default_flow_style=False, indent=2)

print("Created conservative_test.yaml with:")
print(f"- 1 story")
print(f"- 5 features per neuron max")
print(f"- 0.1 threshold (very high)")
print(f"- 50 samples")
print()
print("Run with: python compare_with_config.py --config conservative_test.yaml")