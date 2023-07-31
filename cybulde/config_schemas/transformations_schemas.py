from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class Transformation:
    _target_: str = MISSING


@dataclass
class HuggingFaceTokenizationTransformationConfig:
    _target_: str = "cybulde.data_modules.transformations.HuggingFaceTokenizationTransformation"
    pretrained_tokenizer_name_or_path: str = MISSING
    max_sequence_length: int = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="text_classification_data_module_schema",
        group="tasks/data_module/transformation",
        node=HuggingFaceTokenizationTransformationConfig,
    )
