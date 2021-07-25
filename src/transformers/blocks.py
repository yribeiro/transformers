import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """
    Module that added two tensors of the same size, normalises the layer and returns.
    """

    def __init__(self, layer_size: int):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(layer_size)

    def forward(self, inp1: torch.Tensor, inp2: torch.Tensor) -> torch.Tensor:
        """
        Runs forward inference using the block.

        :param inp1: Tensor 1 in addition.
        :param inp2: Tensor 2 in addition.
        :return Torch tensor that has summed the two inputs and normalised the layer values.
        """
        x = inp1 + inp2
        return self.layer_norm(x)
