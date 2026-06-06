from collections.abc import Iterator
from itertools import islice
from tqdm import tqdm
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore
from einops import rearrange
from model_diffing.data.shuffle import batch_shuffle_tensor_iterator_BX
from model_diffing.scripts.config_common import SequenceIteratorConfig
from model_diffing.data.token_loader import TokenSequenceLoader, TokensSequenceBatch
from model_diffing.data.model_hookpoint_dataloader import ScaledModelHookpointActivationsDataloader
from model_diffing.data.activation_harvester import ActivationsHarvester
from sleepers.scripts.config_common import SleeperDataConfig
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X
from transformer_lens import HookedTransformer


class SleeperTokenSequenceLoader(TokenSequenceLoader):
    SLEEPER_HF_DATASET = "mars-jason-25/tiny_stories_instruct_sleeper_data"
    #SLEEPER_HF_DATASET = "mars-jason-25/processed_dolphin_IHY_sleeper_distilled_dataset"
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        include_sleeper_data: bool,
        cache_dir: str | None = None,
        validation: bool = False,
        sequence_length: int = 128,
        shuffle_buffer_size: int = 1024,
        batch_size: int = 16,
        hf_dataset_name: str | None = None,
        text_field: str = "text",
    ):
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length
        self._include_sleeper_data = include_sleeper_data
        self._validation = validation
        self._shuffle_buffer_size = shuffle_buffer_size
        self._batch_size = batch_size
        self._hf_dataset_name = hf_dataset_name or self.SLEEPER_HF_DATASET
        self._text_field = text_field

    def _get_sequence_iterator(self) -> Iterator[torch.Tensor]:
        text_dataset = load_dataset(
            self._hf_dataset_name, streaming=True, cache_dir=self._cache_dir, split=("test" if self._validation else "train")
        )

        for example in text_dataset:
            # Handle sleeper-specific filtering only for sleeper datasets
            if self._hf_dataset_name == self.SLEEPER_HF_DATASET:
                if not self._include_sleeper_data and not example["is_training"]:
                    continue
            
            example = cast(dict[str, Any], example)
            tokeniser_result = self._tokenizer(example[self._text_field])
            seq_tokens_S = torch.tensor(tokeniser_result["input_ids"])
            assert len(seq_tokens_S.shape) == 1, f"seq_tokens_S.shape should be 1D but was {seq_tokens_S.shape}"
            if len(seq_tokens_S) < self._sequence_length:
                continue
            else:
                yield seq_tokens_S[0 : self._sequence_length]

    # TODO make cached_property like HuggingfaceTextDatasetTokenSequenceLoader?
    def get_sequences_batch_iterator(self) -> Iterator[TokensSequenceBatch]:
        # then, shuffle this iterator (doesn't do much but easier to keep similar to other loaders)
        # this shuffler returns batches, hence (B, S)
        for tokens_BS in batch_shuffle_tensor_iterator_BX(
            tensor_iterator_X=self._get_sequence_iterator(),
            shuffle_buffer_size=self._shuffle_buffer_size,
            yield_batch_size=self._batch_size,
        ):
            yield TokensSequenceBatch(
                tokens_BS=tokens_BS,
                special_tokens_mask_BS=torch.zeros(tokens_BS.shape, dtype=torch.bool),
            )

    def num_batches(self) -> int | None:
        return None


def build_tokens_sequence_loader(
    cfg: SequenceIteratorConfig,
    cache_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> TokenSequenceLoader:
    if cfg.classname == "SleeperTokenSequenceLoader":
        if cfg.kwargs is None:
            raise ValueError("kwargs must be provided")
        return SleeperTokenSequenceLoader(
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            batch_size=batch_size,
            **cfg.kwargs,
        )

    raise ValueError(f"Unknown tokens sequence iterator config name: {cfg}")



@torch.no_grad()
def estimate_mean_SXD(
    dataloader_BSXD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches_for_mean_estimate: int,
) -> torch.Tensor:
    mean_samples = []

    for batch_BSXD in tqdm(
        islice(dataloader_BSXD, n_batches_for_mean_estimate),
        desc="Estimating mean",
        total=n_batches_for_mean_estimate,
    ):
        batch_BSXD = batch_BSXD.to(device)
        means_SXD = batch_BSXD.mean(dim=0)
        mean_samples.append(means_SXD)

    mean_samples_MSXD = torch.stack(mean_samples, dim=0)
    mean_SXD = mean_samples_MSXD.mean(dim=0)
    return mean_SXD


class SequencePositionMeanRemovingScaledModelHookpointActivationsDataloader(ScaledModelHookpointActivationsDataloader):
    def __init__(
        self,
        token_sequence_loader: TokenSequenceLoader,
        activations_harvester: ActivationsHarvester,
        activations_shuffle_buffer_size: int,
        yield_batch_size: int,
        device: torch.device,
        n_batches_for_norm_estimate: int,
        n_batches_for_mean_estimate: int
    ):
        self._token_sequence_loader = token_sequence_loader
        self._activations_harvester = activations_harvester
        self._activations_shuffle_buffer_size = activations_shuffle_buffer_size
        self._yield_batch_size = yield_batch_size

        self._mean_SMPD = estimate_mean_SXD(
            self._activations_iterator_BSMPD(),
            device,
            n_batches_for_mean_estimate,
        )

        # important note: using the unscaled iterator but with means removed
        self._norm_scaling_factors_MP = estimate_norm_scaling_factor_X(
            self._shuffled_raw_activations_iterator_BMPD,
            device,
            n_batches_for_norm_estimate,
        )
    
    @torch.no_grad()
    def _activations_iterator_BSMPD(self) -> Iterator[torch.Tensor]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMPD = self._activations_harvester.get_activations_BSMPD(seq.tokens_BS)
            yield activations_BSMPD

    @torch.no_grad()
    def _activations_iterator_MPD(self) -> Iterator[torch.Tensor]:
        for seq in self._token_sequence_loader.get_sequences_batch_iterator():
            activations_BSMPD = self._activations_harvester.get_activations_BSMPD(seq.tokens_BS)

            activations_BSMPD_mean_removed = activations_BSMPD - self._mean_SMPD.unsqueeze(0)

            activations_BsMPD = rearrange(activations_BSMPD_mean_removed, "b s m p d -> (b s) m p d")
            special_tokens_mask_Bs = rearrange(seq.special_tokens_mask_BS, "b s -> (b s)")
            activations_BsMPD = activations_BsMPD[~special_tokens_mask_Bs]

            yield from activations_BsMPD

def build_dataloader(
    cfg: SleeperDataConfig,
    llms: list[HookedTransformer],
    hookpoints: list[str],
    batch_size: int,
    cache_dir: str,
    device: torch.device,
) -> ScaledModelHookpointActivationsDataloader:
    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = build_tokens_sequence_loader(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=cfg.activations_harvester.harvesting_batch_size,
    )

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        hookpoints=hookpoints,
    )

    activations_dataloader = ScaledModelHookpointActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        device=device,
        n_batches_for_norm_estimate=cfg.n_batches_for_norm_estimate,
    )

    return activations_dataloader



def build_mean_removing_dataloader(
    cfg: SleeperDataConfig,
    llms: list[HookedTransformer],
    hookpoints: list[str],
    batch_size: int,
    cache_dir: str,
    device: torch.device,
) -> ScaledModelHookpointActivationsDataloader:
    tokenizer = llms[0].tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError("Tokenizer is not a PreTrainedTokenizerBase")

    # first, get an iterator over sequences of tokens
    token_sequence_loader = build_tokens_sequence_loader(
        cfg=cfg.sequence_iterator,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        batch_size=cfg.activations_harvester.harvesting_batch_size,
    )

    # then, run these sequences through the model to get activations
    activations_harvester = ActivationsHarvester(
        llms=llms,
        hookpoints=hookpoints,
    )

    activations_dataloader = SequencePositionMeanRemovingScaledModelHookpointActivationsDataloader(
        token_sequence_loader=token_sequence_loader,
        activations_harvester=activations_harvester,
        activations_shuffle_buffer_size=cfg.activations_shuffle_buffer_size,
        yield_batch_size=batch_size,
        device=device,
        n_batches_for_norm_estimate=cfg.n_batches_for_norm_estimate,
        n_batches_for_mean_estimate=cfg.n_batches_for_mean_estimate,
    )

    return activations_dataloader
