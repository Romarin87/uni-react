"""Custom samplers for deterministic and resumable geometric loading."""

import itertools
from typing import Iterator

import torch
from torch.utils.data import Sampler


class EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch random sampler for single-process training."""

    def __init__(self, data_source, seed: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class OffsetSampler(Sampler[int]):
    """Wrap an underlying sampler and skip a prefix of indices."""

    def __init__(self, base_sampler: Sampler[int]) -> None:
        self.base_sampler = base_sampler
        self._skip = 0

    def set_skip(self, skip_samples: int) -> None:
        self._skip = max(0, int(skip_samples))

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[int]:
        return itertools.islice(iter(self.base_sampler), self._skip, None)

    def __len__(self) -> int:
        return max(0, len(self.base_sampler) - self._skip)
