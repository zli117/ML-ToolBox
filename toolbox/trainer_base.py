from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple, List, Optional, Any

import torch
from torch import nn
from torch.optim import Optimizer

from toolbox.callbacks.callback_base import CallBack
from toolbox.metrics.metrics_base import Metrics
from toolbox.trackable import Trackable, serialize_list, deserialize_state, \
    deserialize_list
from toolbox.tracked_data_loader import TrackedDataLoader


@dataclass
class Trainer(Trackable, ABC):
    model: nn.Module
    train_loader: TrackedDataLoader
    valid_loader: TrackedDataLoader
    opt_class: Type
    opt_config: dict
    device: torch.device
    save_optimizer: bool = True
    progress_bar_size: int = field(default=20)
    call_backs: List[CallBack] = field(default_factory=list)
    metrics: List[Metrics] = field(default_factory=list)
    _terminate: bool = field(init=False, default=False)
    _curr_epochs: int = field(init=False, default=0)
    _curr_steps: int = field(init=False, default=0)
    _optimizer: Optional[Optimizer] = field(init=False, default=None)

    def __post_init__(self):
        self.model.to(self.device)

    def stop(self) -> None:
        self._terminate = True

    def add_metrics(self, metrics: Metrics) -> None:
        self.metrics.append(metrics)

    def serialize(self) -> Dict[str, Any]:
        serialized = super().serialize()
        self.model.cpu()
        serialized['model'] = (self.model.__class__, self.model.state_dict())
        self.model.to(self.device)
        if self.save_optimizer:
            serialized['_optimizer'] = self._optimizer.state_dict()
        serialized = self.serialize_trackable_attrs(serialized,
                                                    ['train_loader',
                                                     'valid_loader'])
        serialized['call_backs'] = serialize_list(self.call_backs)
        serialized['metrics'] = serialize_list(self.metrics)
        serialized = self.serialize_plain_attrs(serialized,
                                                ['opt_class',
                                                 'opt_config',
                                                 'device',
                                                 'progress_bar_size',
                                                 '_curr_epochs',
                                                 '_curr_steps'])
        return serialized

    @staticmethod
    def deserialize(state: Dict[str, Any], strict: bool = False,
                    model: Optional[nn.Module] = None) -> 'Trainer':
        """
        Load the Trainer from a serialized dictionary
        Args:
            state: The state dictionary
            strict: If false will ignore missing attributes
            model: User provided model instance. If None, then an instance will
            be created without argument

        Returns:
            Loaded Trainer
        """
        train_loader = deserialize_state(state['train_loader'],
                                         cast_to=TrackedDataLoader)
        valid_loader = deserialize_state(state['valid_loader'],
                                         cast_to=TrackedDataLoader)
        call_backs = deserialize_list(state['call_backs'], cast_to=CallBack)
        metrics = deserialize_list(state['metrics'], cast_to=Metrics)
        opt_class = state['opt_class']
        opt_config = state['opt_config']
        device = state['device']
        progress_bar_size = state['progress_bar_size']
        _curr_epochs = state['_curr_epochs']
        _curr_steps = state['_curr_steps']
        model_cls, model_state = state['model']
        if model is None:
            model = model_cls()
        model.load_state_dict(model_state, strict=strict)
        save_optimizer = False
        _optimizer = None
        if '_optimizer' in state:
            _optimizer = opt_class(
                filter(lambda p: p.requires_grad, model.parameters()),
                **opt_config)
            # Manually moving optimizer state to device
            # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
            for state in _optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            save_optimizer = True
        trainer = Trainer(model, train_loader, valid_loader, opt_class,
                          opt_config, device, save_optimizer, progress_bar_size,
                          call_backs, metrics)
        trainer._curr_epochs = _curr_epochs
        trainer._curr_steps = _curr_steps
        if _optimizer is not None:
            trainer._optimizer = _optimizer
        return trainer

    @abstractmethod
    def parse_train_batch(self, batch: Dict[str, torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse train batch to input and target
        Args:
            batch: The batch from the data loader

        Returns:
            input and target for this batch
        """

    def parse_valid_batch(self, batch: Dict[str, torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse validation batch to input and target
        Args:
            batch: The batch from the data loader

        Returns:
            input and target for this batch
        """
        return self.parse_train_batch(batch)

    @abstractmethod
    def loss_fn(self, output: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Loss function used in train and validation
        Args:
            output: The output from the model
            target: The target

        Returns:
            The loss
        """

    @abstractmethod
    def train(self, epochs: int) -> nn.Module:
        """
        Train function
        Args:
            epochs: Now many epochs to train

        Returns:
            Trained model
        """
