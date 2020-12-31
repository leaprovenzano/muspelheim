from typing import Tuple
import torch
from torch import Tensor


def onehot_inputs_and_targets(inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
    expanded_targets = targets.unsqueeze(-1)
    fill = inputs.new_ones(expanded_targets.shape)
    pred = torch.zeros_like(inputs).scatter_add(-1, inputs.argmax(dim=-1, keepdim=True), fill)
    tru = torch.zeros_like(inputs).scatter_add(-1, expanded_targets, fill)
    return pred, tru
