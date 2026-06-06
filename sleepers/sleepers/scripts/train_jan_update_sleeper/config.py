from pathlib import Path

from pydantic import BaseModel

from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig
from model_diffing.scripts.train_jan_update_crosscoder.config import JumpReLUConfig
from sleepers.scripts.config_common import SleeperDataConfig


class JanUpdateCrosscoderConfig(BaseModel):
    hidden_dim: int
    jumprelu: JumpReLUConfig = JumpReLUConfig()
    ft_init_checkpt_folder: Path | None = None
    ft_init_checkpt_step: int | None = None
    ft_init_wandb_run_id: str | None = None


class JanUpdateTrainConfig(BaseTrainConfig):
    c: float = 4.0
    initial_lambda_s: float = 0.0
    final_lambda_s: float = 20.0
    lambda_p: float = 3e-6
    


class JanUpdateExperimentConfig(BaseExperimentConfig):
    data: SleeperDataConfig
    crosscoder: JanUpdateCrosscoderConfig
    train: JanUpdateTrainConfig
    hookpoints: list[str]