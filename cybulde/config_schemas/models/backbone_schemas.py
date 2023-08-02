from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.config_schemas.models.transformation_schemas import (
    CustomHuggingFaceTokenizationTransformationConfig,
    TransformationConfig,
)
from cybulde.utils.mixins import LoggableParamsMixin


@dataclass
class BackboneConfig(LoggableParamsMixin):
    _target_: str = MISSING
    transformation: TransformationConfig = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_: str = "cybulde.models.backbones.HuggingFaceBackbone"
    pretrained_model_name_or_path: str = MISSING
    pretrained: bool = False

    def loggable_params(self) -> list[str]:
        return super().loggable_params() + ["pretrained_model_name_or_path", "pretrained"]


@dataclass
class BertTinyHuggingFaceBackboneConfig(HuggingFaceBackboneConfig):
    pretrained_model_name_or_path: str = "prajjwal1/bert-tiny"
    transformation: TransformationConfig = CustomHuggingFaceTokenizationTransformationConfig()


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="hugging_face_backbone_schema",
        group="tasks/lightning_module/model/backbone",
        node=HuggingFaceBackboneConfig,
    )

    cs.store(
        name="test_backbone_config",
        node=BertTinyHuggingFaceBackboneConfig,
    )
