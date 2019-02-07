from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, cast

import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler

from .trackable import Trackable, deserialize_state
from .tracked_dataset import TrackedDataset


class _Sampler(Sampler):

    def __init__(self,
                 sampler: Sampler,
                 index_specified: List[int] = None,
                 skip_index: int = 0):
        super().__init__(None)
        self._sampler = sampler
        self._indices = index_specified
        if self._indices is None:
            self._indices = list(iter(self._sampler))
        self._skip_index = skip_index
        assert 0 <= self._skip_index <= len(self._indices)

    @property
    def indices(self):
        return self.indices

    def __iter__(self):
        iterator = iter(self._indices[self._skip_index:])
        self._indices = list(iter(self._sampler))
        self._skip_index = 0
        return iterator

    def __len__(self):
        return len(self._indices) - self._skip_index


@dataclass
class TrackedDataLoader(Trackable):
    dataset: TrackedDataset
    loader_config: Dict[str, Any]
    step_counter: int = 0
    init_index: Optional[List[int]] = None

    def __post_init__(self):
        assert self.step_counter >= 0 and not (self.init_index is None and
                                               self.step_counter != 0)
        provided_sampler = self.get_sampler()
        assert len(self.dataset) == len(provided_sampler)
        indices = (self.init_index[self.step_counter:]
                   if self.init_index else None)
        sampler = _Sampler(provided_sampler, indices)
        self._data_loader = DataLoader(
            self.dataset, sampler=sampler, **self.loader_config)

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized = self.serialize_trackable_attrs(serialized, ['dataset'])
        # TODO: Deal with the case where config contains trackable
        return self.serialize_plain_attrs(
            serialized, ['loader_config', 'step_counter', 'init_index'])

    @staticmethod
    def deserialize(state: dict, strict: bool = False) -> 'TrackedDataLoader':
        dataset: TrackedDataset = deserialize_state(state['dataset'])
        loader_config: Dict[str, Any] = state['loader_config']
        step_counter: int = state['step_counter']
        init_index: List[int] = state['init_index']
        # pytype: disable=wrong-arg-count
        return TrackedDataLoader(dataset, loader_config, step_counter,
                                 init_index)
        # pytype: enable=wrong-arg-count

    def get_sampler(self) -> Sampler:
        """
        Factory function for creating a sampler. Default is RandomSampler
        Returns:
            The sampler
        """
        return RandomSampler(self.dataset)

    def next_epoch(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        self.step_counter %= len(self.dataset)
        assert self.step_counter + len(self._data_loader) == len(self.dataset)

        for batch in self._data_loader:
            self.step_counter += 1
            yield cast(Dict[str, torch.Tensor], batch)

    def __len__(self):
        return len(self.dataset)
