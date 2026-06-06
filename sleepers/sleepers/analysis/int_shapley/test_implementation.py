#!/usr/bin/env python3
"""
Test script for Shapley-Taylor interaction implementation.
Run this to verify compilation and test shapes on a small sample.
"""

import sys
import os
import torch
from datasets import load_dataset

# Add paths to import from the sleepers codebase
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from sleepers.scripts.llms import build_llm_lora
from sleepers.scripts.utils import load_crosscoder_from_wandb
from shapley_interactions import test_shapes_small_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """Run the test with a small sample."""
    print("Loading dataset...")
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    dataset = dataset.filter(lambda x: x['is_training'] == True)
    
    print("Loading LLM...")
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    print("Loading crosscoder...")
    # Using the crosscoder mentioned in feat_ints.py
    crosscoder_name = "86u64trx"  # l=0, bias=True, base XC
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", crosscoder_name, "../../.wandb_artifacts", DEVICE
    )
    
    print(f"Testing on device: {DEVICE}")
    print(f"LLM d_mlp: {llm.cfg.d_mlp}")
    print(f"Crosscoder hidden dim: {crosscoder.W_dec_HXD.shape[0]}")
    
    # Run the test
    try:
        result, test_info = test_shapes_small_sample(
            dataset=dataset,
            llm=llm,
            crosscoder=crosscoder,
            layer=0  # Test layer 0
        )
        
        print("\n🎉 Test completed successfully!")
        
        # Save the small test result for inspection
        output_path = os.path.join(os.path.dirname(__file__), "test_result_small.pt")
        torch.save({
            'interactions': result,
            'test_info': test_info
        }, output_path)
        print(f"Test result saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)