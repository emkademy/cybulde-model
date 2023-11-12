from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.config_schemas.base_schemas import LightningModuleConfig


@dataclass
class EvaluationLightningModuleConfig(LightningModuleConfig):
    _target_: str = MISSING
    _partial_: bool = False

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class PartialEvaluationLightningModuleConfig(EvaluationLightningModuleConfig):
    _partial_: bool = True


@dataclass
class BinaryTextEvaluationLightningModuleConfig(PartialEvaluationLightningModuleConfig):
    _target_: str = "cybulde.evaluation.lightning_modules.binary_text_evaluation.BinaryTextEvaluationLightningModule"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classification_prediction_lightning_module_schema",
        group="tasks/lightning_module",
        node=BinaryTextEvaluationLightningModuleConfig,
    )
