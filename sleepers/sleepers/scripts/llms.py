from typing import cast

import torch
from transformer_lens import HookedTransformer  # type: ignore
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

from transformers import AutoModelForCausalLM
from peft import PeftModel

from model_diffing.scripts.config_common import LLMConfig


def build_llm_lora(base_model_repo: str, lora_model_repo: str, cache_dir: str, device: torch.device, dtype: str) -> HookedTransformer:
    '''
    Create a hooked transformer model from a base model and a LoRA finetuned model.
    '''
    base_model = AutoModelForCausalLM.from_pretrained(base_model_repo)
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_repo,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    lora_model_merged = lora_model.merge_and_unload()
    hooked_model = HookedTransformer.from_pretrained(
        base_model_repo, 
        hf_model=lora_model_merged,
        cache_dir=cache_dir,
        dtype=dtype,
    ).to(device)
    return hooked_model


def load_model_with_tl_check(model_name: str, cache_dir: str = None, device: torch.device = None, dtype: str = "float32") -> HookedTransformer:
    """
    Load a model with TransformerLens, automatically checking compatibility and converting if needed.
    
    Args:
        model_name: HuggingFace model name or TransformerLens model name
        cache_dir: Cache directory for model downloads
        device: Device to load model on
        dtype: Data type for model weights
        
    Returns:
        HookedTransformer: The loaded model
    """
    # Check if model is directly supported by TransformerLens
    if model_name in OFFICIAL_MODEL_NAMES:
        print(f"Loading {model_name} directly from TransformerLens...")
        return HookedTransformer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype
        )
    
    # If not officially supported, try to load from HuggingFace and convert
    print(f"Model {model_name} not in TransformerLens official list. Attempting HuggingFace conversion...")
    try:
        # Load the HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Try to convert it to HookedTransformer
        # We need to find a base model name that TransformerLens can use for architecture
        base_model_name = _infer_base_model_name(model_name)
        
        return HookedTransformer.from_pretrained(
            base_model_name,
            hf_model=hf_model,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype
        )
    except Exception as e:
        raise ValueError(f"Failed to load model {model_name}. Error: {e}")


def _infer_base_model_name(model_name: str) -> str:
    """
    Infer the base TransformerLens model name from a HuggingFace model name.
    This is used for architecture compatibility when converting custom models.
    """
    model_lower = model_name.lower()
    
    # Common model mappings
    if "gpt2" in model_lower:
        return "gpt2"
    elif "tinystories" in model_lower:
        # For TinyStories models, use gpt2 as base architecture
        return "gpt2"
    elif "pythia" in model_lower:
        return "EleutherAI/pythia-160m"  # Use smallest Pythia as base
    elif "llama" in model_lower:
        return "meta-llama/Llama-2-7b-hf"
    elif "mistral" in model_lower:
        return "mistralai/Mistral-7B-v0.1"
    else:
        # Default to gpt2 for unknown models
        print(f"Warning: Unknown model type {model_name}, defaulting to gpt2 architecture")
        return "gpt2"