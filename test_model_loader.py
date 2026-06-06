#!/usr/bin/env python3
"""
Test script for the model loader with TL compatibility check.
Tests loading the GPT2-124M TinyStories model from the config.
"""

import torch
import sys
import os

# Add the sleepers module to path
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers')

from sleepers.scripts.llms import load_model_with_tl_check

def test_model_loader():
    """Test the model loader with the GPT2-124M TinyStories model."""
    
    model_name = "DarwinAnim8or/gpt2-124M-tinystories"
    
    print(f"Testing model loader with: {model_name}")
    print("=" * 50)
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Test the model loader
        model = load_model_with_tl_check(
            model_name=model_name,
            cache_dir="./cache",
            device=device,
            dtype="float32"
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model config: {model.cfg}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test a simple forward pass
        print("\nTesting forward pass...")
        test_input = torch.randint(0, 1000, (1, 10)).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loader()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)