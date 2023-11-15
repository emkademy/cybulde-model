from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.utils.mixins import LoggableParamsMixin


@dataclass
class TransformationConfig(LoggableParamsMixin):
    _target_: str = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class HuggingFaceTokenizationTransformationConfig(TransformationConfig):
    _target_: str = "cybulde.models.transformations.HuggingFaceTokenizationTransformation"
    pretrained_tokenizer_name_or_path: str = MISSING
    max_sequence_length: int = MISSING

    def loggable_params(self) -> list[str]:
        return super().loggable_params() + ["pretrained_tokenizer_name_or_path", "max_sequence_length"]


@dataclass
class CustomHuggingFaceTokenizationTransformationConfig(HuggingFaceTokenizationTransformationConfig):
    pretrained_tokenizer_name_or_path: str = "gs://emkademy/cybulde/data/processed/rebalanced_splits/trained_tokenizer"
    max_sequence_length: int = 200


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="text_classification_data_module_schema",
        group="tasks/data_module/transformation",
        node=HuggingFaceTokenizationTransformationConfig,
    )
