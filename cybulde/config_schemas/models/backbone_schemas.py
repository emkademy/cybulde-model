from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.config_schemas.transformations_schemas import TransformationConfig


@dataclass
class BackboneConfig:
    _target_: str = MISSING
    transformation: TransformationConfig = MISSING


@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_: str = "cybulde.models.backbones.HuggingFaceBackbone"
    pretrained_model_name_or_path: str = MISSING
    pretrained: bool = False


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="hugging_face_backbone_schema",
        group="tasks/lightning_module/model/backbone",
        node=HuggingFaceBackboneConfig,
    )
