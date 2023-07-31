from typing import Optional

from torch import Tensor, nn
from transformers import BatchEncoding

from cybulde.models.adapters import Adapter
from cybulde.models.backbones import Backbone
from cybulde.models.heads import Head


class Model(nn.Module):
    pass


class BinaryTextClassificationModel(Model):
    def __init__(self, backbone: Backbone, head: Head, adapter: Optional[Adapter]) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, encodings: BatchEncoding) -> Tensor:
        output = self.backbone(encodings)
        if self.adapter is not None:
            output = self.adapter(output)
        output = self.head(output)
        return output
