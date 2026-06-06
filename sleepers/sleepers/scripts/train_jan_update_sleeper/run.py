import fire  # type: ignore
import torch
import wandb
from pathlib import Path
from model_diffing.models.crosscoder import (
    AcausalCrosscoder,
)
from model_diffing.models.activations.jumprelu import JumpReLUActivation
from model_diffing.log import logger
from sleepers.scripts.train_jan_update_sleeper.config import JanUpdateExperimentConfig
from sleepers.scripts.train_jan_update_sleeper.trainer import JanUpdateSleeperTrainer
from model_diffing.scripts.utils import build_wandb_run, build_optimizer
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.utils import get_device
from sleepers.scripts.llms import build_llm_lora
from sleepers.data.dataloader import build_dataloader, build_mean_removing_dataloader
from model_diffing.scripts.train_jan_update_crosscoder.run import JanUpdateInitStrategy
from model_diffing.data.model_hookpoint_dataloader import estimate_norm_scaling_factor_X
from sleepers.scripts.utils import calculate_fvu_X

def download_checkpoint_from_wandb(run_id: str, entity: str, project: str, step: int, download_dir: Path = Path(".checkpoints")) -> Path:
    """Download checkpoint from wandb run and return the path to the specific step checkpoint."""
    api = wandb.Api()
    artifact_name = f"model-checkpoint_run-{run_id}:latest"
    
    try:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
        
        # Create download directory
        run_download_dir = download_dir / f"wandb_{run_id}"
        run_download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download artifact
        artifact.download(root=str(run_download_dir))
        
        # The checkpoint is in the model subdirectory
        checkpoint_path = run_download_dir / "model" / f"epoch_0_step_{step}"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint step {step} not found in downloaded artifact. Available files: {list((run_download_dir / 'model').glob('*'))}")
        
        logger.info(f"Downloaded checkpoint from wandb run {run_id} to {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Failed to download checkpoint from wandb run {run_id}: {e}")
        raise

def build_jan_update_sleeper_trainer(cfg: JanUpdateExperimentConfig) -> JanUpdateSleeperTrainer:
    device = get_device()

    llms = [build_llm_lora(
        llm.name,
        llm.lora_name,
        cfg.cache_dir,
        device,
        cfg.data.activations_harvester.inference_dtype
        ) for llm in cfg.data.activations_harvester.llms]

    cfg.data.sequence_iterator.kwargs["validation"] = False
    dataloader = build_mean_removing_dataloader(
        cfg.data,
        llms,
        cfg.hookpoints,
        cfg.train.batch_size,
        cfg.cache_dir,
        device,
    )

    # means = dataloader._mean_SMPD.clone()
    # dataloader._mean_SMPD = torch.zeros_like(dataloader._mean_SMPD)
    # # Recalculate norm scaling factors with zero mean
    # dataloader._norm_scaling_factors_MP = estimate_norm_scaling_factor_X(
    #     dataloader._shuffled_raw_activations_iterator_BMPD,
    #     device,
    #     cfg.data.n_batches_for_norm_estimate,
    # )

    n_models = len(llms)
    n_hookpoints = len(cfg.hookpoints)

    # Handle checkpoint loading from local folder or wandb
    checkpoint_path = None
    if cfg.crosscoder.ft_init_checkpt_folder is not None:
        checkpoint_path = cfg.crosscoder.ft_init_checkpt_folder / f"epoch_0_step_{cfg.crosscoder.ft_init_checkpt_step}"
    elif cfg.crosscoder.ft_init_wandb_run_id is not None:
        if cfg.crosscoder.ft_init_checkpt_step is None:
            raise ValueError("ft_init_checkpt_step must be specified when using ft_init_wandb_run_id")
        checkpoint_path = download_checkpoint_from_wandb(
            run_id=cfg.crosscoder.ft_init_wandb_run_id,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            step=cfg.crosscoder.ft_init_checkpt_step
        )
    
    if checkpoint_path is not None:
        crosscoder = AcausalCrosscoder.load(checkpoint_path, device=device)
        crosscoder = crosscoder.to(device)  # Ensure model is on correct device
        norm_scaling_factors_X = crosscoder.unfold_activation_scaling_from_weights_()
        dataloader.norm_scaling_factors_X = norm_scaling_factors_X
    else:
        crosscoder = AcausalCrosscoder(
            crosscoding_dims=(n_models, n_hookpoints),
            d_model=llms[0].cfg.d_model,
            hidden_dim=cfg.crosscoder.hidden_dim,
            init_strategy=JanUpdateInitStrategy(
                activations_iterator_BXD=dataloader.get_shuffled_activations_iterator_BMPD(),
                initial_approx_firing_pct=0.5,
                n_examples_to_sample=10_000,
            ),
            hidden_activation=JumpReLUActivation(
                size=cfg.crosscoder.hidden_dim,
                bandwidth=cfg.crosscoder.jumprelu.bandwidth,
                log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
                backprop_through_input=cfg.crosscoder.jumprelu.backprop_through_jumprelu_input
            ),
        )
    crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    torch.save(dataloader._mean_SMPD, cfg.save_dir / "dataloader_means.pt")
    artifact = wandb.Artifact(f"dataloader-means_run-{wandb_run.id}", type="dataset")
    artifact.add_file(cfg.save_dir / "dataloader_means.pt")
    wandb.log_artifact(artifact)

    trainer = JanUpdateSleeperTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
    )
    # Modify optimizer to exclude decoder bias from optimization
    # Get all parameters except decoder bias
    params_to_optimize = []
    for name, param in crosscoder.named_parameters():
        if name != "b_dec_XD":
            params_to_optimize.append(param)
    # Replace the optimizer with one that only optimizes the selected parameters
    trainer.optimizer = build_optimizer(cfg.train.optimizer, params_to_optimize)
    logger.info("Modified optimizer to exclude decoder bias from optimization")
    # Note decoder bias is initialized to 0
    assert crosscoder.b_dec_XD.norm() == 0

    # test = torch.randn(128, 1, 12, 768)
    # test_enc = crosscoder._encode_BH(test)
    # test_dec = crosscoder._decode_BXD(test_enc)
    # print(test_enc, test_dec)
    # print(trainer._get_loss(test))

    # dataloader._mean_SMPD = means.clone()
    # crosscoder.b_enc_H.data += einsum(means[0], crosscoder.W_enc_XDH, "m p d, m p d h -> h")
    # crosscoder.b_dec_XD.data -= means[0]

    # test -= means
    # test_enc = crosscoder._encode_BH(test)
    # test_dec = crosscoder._decode_BXD(test_enc)
    # print(test_enc, test_dec, test_dec+means.unsqueeze(0))
    # print(trainer._get_loss(test))

    return trainer
    


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_jan_update_sleeper_trainer, JanUpdateExperimentConfig))
