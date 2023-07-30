
from torch import nn
from transformers import AutoConfig, AutoModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput

from cybulde.utils.io_utils import translate_gcs_dir_to_local


class Backbone(nn.Module):
    pass


class HuggingFaceBackbone(Backbone):
    def __init__(self, pretrained_model_name_or_path: str, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = self.get_backbone(pretrained_model_name_or_path, pretrained)

    def forward(self, encodings: BatchEncoding) -> BaseModelOutput:
        output: BaseModelOutput = self.backbone(**encodings)
        return output

    def get_backbone(self, pretrained_model_name_or_path: str, pretrained: bool) -> nn.Module:
        path = translate_gcs_dir_to_local(pretrained_model_name_or_path)
        config = AutoConfig.from_pretrained(path)
        if pretrained:
            return AutoModel.from_pretrained(path, config=config)
        return AutoModel.from_config(config)
