from pathlib import Path

import fire  # type: ignore
import yaml  # type: ignore
import torch
import wandb
from einops import rearrange

from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_l1_crosscoder.trainer import AnthropicTransposeInit
from model_diffing.models.activations.topk import TopkActivation, BatchTopkActivation
from sleepers.scripts.train_topk_sleeper.config import TopKExperimentConfig
from sleepers.scripts.train_topk_sleeper.trainer import TopKTrainer
from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device, size_human_readable
from sleepers.scripts.llms import build_llm_lora, load_model_with_tl_check
from transformer_lens import HookedTransformer
from sleepers.data.dataloader import build_dataloader
from sleepers.scripts.utils import sharpness_func
from datetime import datetime

def harvest_pre_pre_bias_acts(
    data_loader: BaseModelHookpointActivationsDataloader,
    W_enc_XDH: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int = 100_000,
) -> torch.Tensor:
    batch_size = data_loader._yield_batch_size

    remainder = n_examples_to_sample % batch_size
    if remainder != 0:
        logger.warning(
            f"n_examples_to_sample {n_examples_to_sample} must be divisible by the batch "
            f"size {batch_size}. Rounding up to the nearest multiple of batch_size."
        )
        # Round up to the nearest multiple of batch_size:
        n_examples_to_sample = (((n_examples_to_sample - remainder) // batch_size) + 1) * batch_size

        logger.info(f"n_examples_to_sample is now {n_examples_to_sample}")

    activations_iterator_BMPD = data_loader.get_shuffled_activations_iterator_BMPD()

    def get_batch_pre_bias_pre_act() -> torch.Tensor:
        # this is essentially the first step of the crosscoder forward pass, but not worth
        # creating a new method for it, just (easily) reimplementing it here
        batch_BMPD = next(activations_iterator_BMPD)
        x_BH = torch.einsum("b m l d, m l d h -> b h", batch_BMPD, W_enc_XDH)
        return x_BH

    first_sample_BH = get_batch_pre_bias_pre_act()
    hidden_size = first_sample_BH.shape[1]

    pre_bias_pre_act_buffer_NH = torch.empty(n_examples_to_sample, hidden_size, device=device)
    logger.info(
        f"pre_bias_pre_act_buffer_NH.shape: {pre_bias_pre_act_buffer_NH.shape}, "
        f"size: {size_human_readable(pre_bias_pre_act_buffer_NH)}"
    )

    pre_bias_pre_act_buffer_NH[:batch_size] = first_sample_BH
    examples_sampled = batch_size

    while examples_sampled < n_examples_to_sample:
        batch_pre_bias_pre_act_BH = get_batch_pre_bias_pre_act()
        pre_bias_pre_act_buffer_NH[examples_sampled : examples_sampled + batch_size] = batch_pre_bias_pre_act_BH
        examples_sampled += batch_size
    return pre_bias_pre_act_buffer_NH

def _compute_b_enc_H(
    data_loader: BaseModelHookpointActivationsDataloader,
    W_enc_XDH: torch.Tensor,
    initial_threshold_H: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int = 50_000,
) -> torch.Tensor:
    print(f"[MEMORY] Before harvesting: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
    
    pre_bias_pre_act_buffer_NH = harvest_pre_pre_bias_acts(data_loader, W_enc_XDH, device, n_examples_to_sample)
    
    print(f"[MEMORY] After harvesting: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
    print(f"[MEMORY] Buffer shape: {pre_bias_pre_act_buffer_NH.shape}, size: {pre_bias_pre_act_buffer_NH.numel() * pre_bias_pre_act_buffer_NH.element_size() / 1e9:.2f}GB")

    # find the threshold for each idx H such that 1/10_000 of the examples are above the threshold
    # Move to CPU to avoid GPU memory issues during quantile computation
    print(f"Computing quantile on CPU to save GPU memory...")
    pre_bias_pre_act_buffer_cpu = pre_bias_pre_act_buffer_NH.cpu()
    torch.cuda.empty_cache()  # Free GPU memory after moving to CPU
    print(f"[MEMORY] After moving to CPU: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
    
    quantile_H = torch.quantile(pre_bias_pre_act_buffer_cpu, 1 - 1 / 10_000, dim=0).to(device)
    del pre_bias_pre_act_buffer_cpu  # Clean up CPU memory
    
    print(f"[MEMORY] After quantile computation: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

    b_enc_H = initial_threshold_H - quantile_H

    return b_enc_H

def build_llm_with_optional_lora(base_model_repo: str, lora_model_repo: str | None, cache_dir: str, device: torch.device, dtype: str) -> HookedTransformer:
    '''
    Create a hooked transformer model from a base model, optionally with LoRA.
    '''
    if lora_model_repo is None:
        # Load base model without LoRA using our smart loader
        hooked_model = load_model_with_tl_check(
            base_model_repo,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )
        return hooked_model
    else:
        # Load with LoRA
        return build_llm_lora(base_model_repo, lora_model_repo, cache_dir, device, dtype)

def update_experiment_name_in_yaml(yaml_path):
    """Update the experiment name to ensure uniqueness while preserving parameter sweep names."""
    
    # Load the existing configuration
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    current_name = config.get("experiment_name", "")
    
    # If it looks like a parameter sweep name (contains "lam" and "dim"), preserve it
    # Otherwise, generate the traditional name format
    if "lam" in current_name and "dim" in current_name:
        # Preserve parameter sweep naming format
        print(f'experiment_name: {current_name}')
        # Don't modify it - let BaseExperimentConfig add timestamp for uniqueness
    else:
        # Use traditional naming format for single runs
        traditional_name = f"lambda_n{config['train']['lam_n']}_beta_n{config['train']['beta_n']}_S"
        config["experiment_name"] = traditional_name
        print(f'updated experiment name: {traditional_name}')
    
    # Save the updated configuration back to the YAML file
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(config, file)

def download_checkpoint_from_wandb_topk(run_id: str, entity: str, project: str, download_dir: Path = Path(".checkpoints")) -> Path:
    """Download checkpoint from wandb run and return the path to the SaveableModule checkpoint."""
    api = wandb.Api()
    artifact_name = f"model-checkpoint_run-{run_id}:latest"
    
    try:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
        
        # Create download directory
        run_download_dir = download_dir / f"wandb_{run_id}"
        run_download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download artifact
        artifact.download(root=str(run_download_dir))
        
        # The checkpoint uses SaveableModule format (model.pt + model_cfg.yaml)
        checkpoint_dir = run_download_dir / "model"
        model_pt_path = checkpoint_dir / "model.pt"
        model_cfg_path = checkpoint_dir / "model_cfg.yaml"
        
        if not model_pt_path.exists() or not model_cfg_path.exists():
            available_files = list(checkpoint_dir.glob('*')) if checkpoint_dir.exists() else []
            raise FileNotFoundError(f"SaveableModule checkpoint files not found. Available files: {available_files}")
        
        logger.info(f"Downloaded checkpoint from wandb run {run_id} to {checkpoint_dir}")
        return checkpoint_dir  # Return the directory containing model.pt and model_cfg.yaml
        
    except Exception as e:
        logger.error(f"Failed to download checkpoint from wandb run {run_id}: {e}")
        raise

def build_trainer(cfg: TopKExperimentConfig) -> TopKTrainer:
    # Set CUDA memory allocator to avoid fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    device = get_device()
    print(f"[MEMORY] Initial: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
    
    #cheeky way to change experiment name
    #cfg.experiment_name = f"lambda_n{cfg.train.lam_n}_beta_n{cfg.train.beta_n}_{cfg.experiment_name}"
    llms = [build_llm_with_optional_lora(
        llm.name,
        llm.lora_name,
        cfg.cache_dir,
        device,
        cfg.data.activations_harvester.inference_dtype
        ) for llm in cfg.data.activations_harvester.llms]
    
    print(f"[MEMORY] After loading LLMs: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")

    # TODO remove validation?
    cfg.data.sequence_iterator.kwargs["validation"] = False
    dataloader = build_dataloader(
        cfg.data,
        llms,
        cfg.hookpoints,
        cfg.train.batch_size,
        cfg.cache_dir,
        device,
    )
    
    print(f"[MEMORY] After building dataloader: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")

    n_models = len(llms)
    n_hookpoints = len(cfg.hookpoints)

    cc = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
        hidden_activation=BatchTopkActivation(k_per_example=cfg.crosscoder.k),
    )
    print(f"[MEMORY] After creating crosscoder: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
    # TODO update to use init strategies properly?

    with torch.no_grad():
        # parameters from the jan update doc

        n = float(n_models * n_hookpoints * llms[0].cfg.d_model)  # n is the size of the input space
        m = float(cfg.crosscoder.hidden_dim)  # m is the size of the hidden space

        # W_dec ~ U(-1/n, 1/n) (from doc)
        cc.W_dec_HXD.uniform_(-1.0 / n, 1.0 / n)

        # For now, assume we're in the X == Y case.
        # Therefore W_enc = (n/m) * W_dec^T
        cc.W_enc_XDH.copy_(
            rearrange(cc.W_dec_HXD, "hidden model layer d_model -> model layer d_model hidden")  #
            * (n / m)
        )
        
        print(f"[MEMORY] After weight initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
        
        # Force cleanup before bias calibration
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        calibrated_b_enc_H = _compute_b_enc_H(
            dataloader,
            cc.W_enc_XDH.to(device),
            torch.nn.Parameter(torch.ones(cfg.crosscoder.hidden_dim) * 0.1).exp().to(device),
            device,
        )
        cc.b_enc_H.copy_(calibrated_b_enc_H)

        # no data-dependent initialization of b_dec
        cc.b_dec_XD.zero_()
        cc.b_dec_XD.requires_grad = False

    crosscoder = cc
    crosscoder = crosscoder.to(device)

    # Handle checkpoint loading from local folder or wandb
    checkpoint_path = None
    if cfg.crosscoder.ft_init_checkpt_folder is not None:
        # For local checkpoints, check if it's epoch-based or SaveableModule format
        if cfg.crosscoder.ft_init_checkpt_epoch is not None:
            # Epoch-based checkpoint (legacy format)
            checkpoint_file = cfg.crosscoder.ft_init_checkpt_folder / f"model_epoch_{cfg.crosscoder.ft_init_checkpt_epoch}.pt"
            print(f"Loading epoch-based checkpoint from {checkpoint_file}")
            state_dict = torch.load(checkpoint_file)
            unit_scaling_factors_X = torch.ones(crosscoder.crosscoding_dims, device=device)
            crosscoder.folded_scaling_factors_X = unit_scaling_factors_X
            crosscoder.load_state_dict(state_dict)
        else:
            # SaveableModule format
            checkpoint_path = cfg.crosscoder.ft_init_checkpt_folder
    elif cfg.crosscoder.ft_init_wandb_run_id is not None:
        checkpoint_path = download_checkpoint_from_wandb_topk(
            run_id=cfg.crosscoder.ft_init_wandb_run_id,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project
        )
    
    # Load using SaveableModule format if checkpoint_path is set
    if checkpoint_path is not None:
        print(f"Loading SaveableModule checkpoint from {checkpoint_path}")
        crosscoder = AcausalCrosscoder.load(checkpoint_path, device=device)
        crosscoder = crosscoder.to(device)  # Ensure model is on correct device
        norm_scaling_factors_X = crosscoder.unfold_activation_scaling_from_weights_()
        dataloader.norm_scaling_factors_X = norm_scaling_factors_X
    
    
    
    wandb_run = build_wandb_run(cfg)# if cfg.wandb else None

    return TopKTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
        llms=llms,
    )


def run_experiment_with_config_update(config_path: str):
    """Run experiment with automatic config update for experiment naming."""
    # Update experiment name in the actual config file being used (preserves parameter sweep names)
    update_experiment_name_in_yaml(config_path)
    
    # Now run the normal experiment flow
    inner_fn = run_exp(build_trainer, TopKExperimentConfig)
    inner_fn(Path(config_path))

if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_experiment_with_config_update)
