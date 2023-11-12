from typing import Optional

import torch.nn.functional as F

from torch import Tensor, nn


class LossFunction(nn.Module):
    pass


class BCEWithLogitsLoss(LossFunction):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor, pos_weight: Optional[Tensor] = None) -> Tensor:
        return F.binary_cross_entropy_with_logits(x, target, reduction=self.reduction, pos_weight=pos_weight)
