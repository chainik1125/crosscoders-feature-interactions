from pathlib import Path

from pydantic import BaseModel

from model_diffing.scripts.config_common import (
    BaseExperimentConfig,
    BaseTrainConfig,
)
from sleepers.scripts.config_common import SleeperDataConfig

class TopKCrosscoderConfig(BaseModel):
    hidden_dim: int
    dec_init_norm: float = 0.1
    k: int
    ft_init_checkpt_folder: Path | None = None
    ft_init_checkpt_epoch: int | None = None
    ft_init_wandb_run_id: str | None = None


class TopKExperimentConfig(BaseExperimentConfig):
    data: SleeperDataConfig
    crosscoder: TopKCrosscoderConfig
    train: BaseTrainConfig
    hookpoints: list[str]