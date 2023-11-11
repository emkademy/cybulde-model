from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.config_schemas.base_schemas import LightningModuleConfig


@dataclass
class PredictionLightningModuleConfig(LightningModuleConfig):
    _target_: str = MISSING
    _partial_: bool = False

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class PartialPredictionLightningModuleConfig(PredictionLightningModuleConfig):
    _partial_: bool = True


@dataclass
class BinaryTextClassificationPredictionLightningModuleConfig(PartialPredictionLightningModuleConfig):
    _target_: str = "cybulde.prediction.lightning_modules.binary_text_classification.BinaryTextClassificationPredictionLightningModule"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classification_prediction_lightning_module_schema",
        group="tasks/lightning_module",
        node=BinaryTextClassificationPredictionLightningModuleConfig,
    )
