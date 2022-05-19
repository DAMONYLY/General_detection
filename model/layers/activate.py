import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):

        return x * (torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

def hard_mish(x, inplace: bool = False) :
    """Implements the HardMish activation function
    Args:
        x: input tensor
    Returns:
        output tensor
    """

    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)

class HardMish(nn.Module):
    """Implements the Had Mish activation module from `"H-Mish" <https://github.com/digantamisra98/H-Mish>`_
    This activation is computed as follows:
    .. math::
        f(x) = \\frac{x}{2} \\cdot \\min(2, \\max(0, x + 2))
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, inplace=self.inplace)