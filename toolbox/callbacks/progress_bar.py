from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import torch

from toolbox.callbacks.callback_base import CallBack
from toolbox.utils.progress_bar import ProgressBar


@dataclass
class ProgressBarCB(CallBack):
    progress_bar_size: int
    _progress_bar: Optional[ProgressBar] = field(init=False, default=None)
    _curr_epoch: int = field(init=False, default=0)

    def on_train_epoch_begin(self, curr_epoch: int, total_epochs: int) -> None:
        self._curr_epoch = curr_epoch
        self._progress_bar = ProgressBar(self.progress_bar_size,
                                         ' loss: %.06f, batch: %d, epoch: %d')

    def on_train_batch_end(self, curr_step: int, total_steps: int,
                           loss: torch.Tensor) -> None:
        if self._progress_bar is not None:
            self._progress_bar.progress(curr_step / total_steps * 100, loss,
                                        self._curr_epoch)

    def on_train_epoch_end(self, curr_epoch: int, total_epochs: int) -> None:
        # Print a new line at the end of each epoch
        print()

    def serialize(self):
        serialized = super().serialize()
        serialized['progress_bar_size'] = self.progress_bar_size

    @staticmethod
    def deserialize(state: Dict[str, Any],
                    strict: bool = False) -> 'ProgressBarCB':
        assert 'progress_bar_size' in state
        progress_bar_cb = ProgressBarCB(state['progress_bar_size'])
        return progress_bar_cb
