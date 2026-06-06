#!/usr/bin/env python3
"""Check model dimensions for pythia-70m vs tinystories-33m."""

from transformer_lens import HookedTransformer

# Load both models to compare dimensions
print("TinyStories-33M dimensions:")
tiny_model = HookedTransformer.from_pretrained("roneneldan/TinyStories-Instruct-33M")
print(f"  d_model: {tiny_model.cfg.d_model}")
print(f"  d_mlp: {tiny_model.cfg.d_mlp}")
print(f"  n_layers: {tiny_model.cfg.n_layers}")

print("\nPythia-70M dimensions:")
pythia_model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m")
print(f"  d_model: {pythia_model.cfg.d_model}")
print(f"  d_mlp: {pythia_model.cfg.d_mlp}")
print(f"  n_layers: {pythia_model.cfg.n_layers}")