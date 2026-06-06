#!/usr/bin/env python3
"""Quick script to check available hookpoints for pythia-70m model."""

from transformer_lens import HookedTransformer

# Load pythia-70m model
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m")

print("Available hookpoints for pythia-70m:")
for name in sorted(model.hook_dict.keys()):
    print(f"  {name}")

print(f"\nModel has {model.cfg.n_layers} layers")
print(f"Model architecture: {model.cfg.model_name}")