from abc import ABC
from typing import Tuple, Type

import torch
from torch import nn
from torch.optim import Optimizer

from toolbox.trackable import Trackable
from toolbox.tracked_data_loader import TrackedDataLoader


class CallBack(Trackable, ABC):

    def on_train_epoch_begin(self, curr_epoch: int, total_epochs: int) -> None:
        """
        Called at the beginning of each training epoch
        Args:
            curr_epoch: The index of current epoch
            total_epochs: Total number of epochs
        """

    def on_train_epoch_end(self, curr_epoch: int, total_epochs: int) -> None:
        """
        Called at the end of each training epoch, but before validation
        Args:
            curr_epoch: The index of current epoch
            total_epochs: The number of total epochs
        """

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
                           metrics: Tuple[torch.Tensor]) -> None:
        """
        Called at the end of a validation batch
        Args:
            current_step: the current step
            total_steps: Total steps for validation
            metrics: The metrics returned by validation metrics function
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

    def on_train_begin(self, model: nn.Module, train_loader: TrackedDataLoader,
                       opt_class: Type[Optimizer], opt_config: dict,
                       epochs: int, device: torch.device,
                       save_optimizer: bool) -> None:
        """
        Called at the beginning of training
        Args:
            model: The model to be trained
            train_loader: Training data loader
            opt_class: The optimizer class
            opt_config: The optimizer config
            epochs: How many epochs to train
            device: Where the model will run
            save_optimizer: Whether save optimizer states or not on check point
        """

    def on_train_end(self, model: nn.Module, epochs: int) -> None:
        """
        Called at the end of training
        Args:
            model: The model trained
            epochs: How many epochs actually elapsed
        """

    def on_exception(self, exception: BaseException) -> None:
        """
        Called when there's an exception during training loop
        Args:
            exception: The exception type
        """
