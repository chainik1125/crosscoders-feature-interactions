from datetime import datetime
from operator import xor
from typing import Any

from model_diffing.scripts.config_common import LLMConfig, ActivationsHarvesterConfig, DataConfig


class LoraLLMConfig(LLMConfig):
    lora_name: str | None = None


class SleeperActivationsHarvesterConfig(ActivationsHarvesterConfig):
    llms: list[LoraLLMConfig]


class SleeperDataConfig(DataConfig):
    activations_harvester: SleeperActivationsHarvesterConfig
    n_batches_for_mean_estimate: int
