from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.config_schemas.models import adapter_schemas, backbone_schemas, head_schemas


@dataclass
class ModelConfig:
    _target_: str = MISSING


@dataclass
class BinaryTextClassificationModelConfig(ModelConfig):
    _target_: str = "cybulde.models.models.BinaryTextClassificationModel"
    backbone: backbone_schemas.BackboneConfig = MISSING
    adapter: Optional[adapter_schemas.AdapterConfig] = None
    head: head_schemas.HeadConfig = MISSING


def setup_config() -> None:
    backbone_schemas.setup_config()
    adapter_schemas.setup_config()
    head_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classification_model_schema",
        group="tasks/lightning_module/model",
        node=BinaryTextClassificationModelConfig,
    )
