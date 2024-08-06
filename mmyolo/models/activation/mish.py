# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.utils import digit_version

if digit_version(torch.__version__) >= digit_version('1.7.0'):
    MODELS.register_module(module=nn.Mish, name='Mish')
else:

    @MODELS.register_module()
    class Mish(nn.Module):
        """Clamp activation layer.

        This activation function is to clamp the feature map value within
        :math:`[min, max]`. More details can be found in ``torch.clamp()``.

        Args:
            min (Number | optional): Lower-bound of the range to be clamped to.
                Default to -1.
            max (Number | optional): Upper-bound of the range to be clamped to.
                Default to 1.
        """

        def forward(self, x) -> torch.Tensor:
            """Applies the Mish activation function, a smooth alternative to
            ReLU."""
            return x * F.softplus(x).tanh()


def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return MODELS.build(cfg)
