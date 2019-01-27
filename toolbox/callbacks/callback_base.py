from abc import ABC
from typing import Tuple, Type

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset

from toolbox.trackable import Trackable


class CallBack(Trackable, ABC):
    def on_train_batch_begin(self, curr_step: int, total_steps: int,
                             batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Called at the start of each train batch
        Args:
            curr_step: Current step index
            total_steps: Total steps in this epoch
            batch: The batch from the data loader. Tuple of input and target
        """

    def on_train_batch_end(self, curr_step: int, total_steps: int,
                           loss: torch.Tensor) -> None:
        """
        Called at the end of each train batch
        Args:
            curr_step: Current step index
            total_steps: Total steps in this epoch
            loss: The loss of this step. A PyTorch Tensor

        """

    def on_valid_batch_begin(self, current_step: int, total_steps: int,
                             batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Called at the beginning of a validation batch
        Args:
            current_step: Current step index
            total_steps: Total steps for validation
            batch: A tuple of input and target
        """

    def on_valid_batch_end(self, current_step: int, total_steps: int,
                           *metrics: torch.Tensor) -> None:
        """
        Called at the end of a validation batch
        Args:
            current_step: the current step
            total_steps: Total steps for validation
            *metrics: The metrics returned by validation metrics function
        """

    def on_pred_batch_begin(self, current_step: int, total_steps: int,
                            batch: torch.Tensor) -> None:
        """
        Called at the beginning of each prediction batch
        Args:
            current_step: Current step
            total_steps: Total steps for prediction
            batch: The batch input
        """

    def on_pred_batch_end(self, current_step: int, total_steps: int,
                          prediction: torch.Tensor) -> None:
        """
        Called at the end of each prediction batch
        Args:
            current_step: Current step
            total_steps: Total steps for prediction
            prediction: The prediction made in this batch
        """

    def on_train_begin(self, model: nn.Module, train_dataset: Dataset,
                       opt_class: Type[Optimizer], opt_config: dict,
                       train_loader_config: dict, epochs: int, gpu: bool,
                       save_optimizer: bool) -> None:
        """
        Called at the beginning of training
        Args:
            model: The model to be trained
            train_dataset: The dataset used for training
            opt_class: The optimizer class
            opt_config: The optimizer config
            train_loader_config: The DataLoader config
            epochs: How many epochs to train
            gpu: Whether use GPU or not
            save_optimizer: Whether save optimizer states or not on check point
        """
