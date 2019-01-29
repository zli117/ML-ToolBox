from typing import Iterable

import torch
from torch import nn


def get_trainable_parameters(model: nn.Module) -> Iterable[torch.Tensor]:
    return filter(lambda p: p.requires_grad, model.parameters())
