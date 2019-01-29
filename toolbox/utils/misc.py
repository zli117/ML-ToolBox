from typing import Union, Optional

import torch
from torch import nn


def cuda(x: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
    return x.cuda() if torch.cuda.is_available() else x


def save_model(model: nn.Module, file_path: str, epoch: int, seed: int,
               step: Optional[int] = 0) -> None:
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'seed': seed,
                'step': step}, file_path)


def load_model(file_path: str, model: nn.Module) -> nn.Module:
    obj = torch.load(file_path)
    model_state = obj['model']
    model.load_state_dict(model_state)
    del model_state
    return model
