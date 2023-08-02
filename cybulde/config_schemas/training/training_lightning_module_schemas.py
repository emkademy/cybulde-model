from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.config_schemas.base_schemas import LightningModuleConfig
from cybulde.config_schemas.models.model_schemas import BertTinyBinaryTextClassificationModelConfig, ModelConfig
from cybulde.config_schemas.training import loss_schemas, optimizer_schemas, scheduler_schemas
from cybulde.utils.mixins import LoggableParamsMixin


@dataclass
class TrainingLightningModuleConfig(LightningModuleConfig, LoggableParamsMixin):
    _target_: str = MISSING
    model: ModelConfig = MISSING
    loss: loss_schemas.LossFunctionConfig = MISSING
    optimizer: optimizer_schemas.OptimizerConfig = MISSING
    scheduler: Optional[scheduler_schemas.LightningSchedulerConfig] = None

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class BinaryTextClassificationTrainingLightningModuleConfig(TrainingLightningModuleConfig):
    _target_: str = (
        "cybulde.training.lightning_modules.binary_text_classification.BinaryTextClassificationTrainingLightningModule"
    )


@dataclass
class CybuldeBinaryTextClassificationTrainingLightningModuleConfig(
    BinaryTextClassificationTrainingLightningModuleConfig
):
    model: ModelConfig = BertTinyBinaryTextClassificationModelConfig()
    loss: loss_schemas.LossFunctionConfig = loss_schemas.BCEWithLogitsLossConfig()
    optimizer: optimizer_schemas.OptimizerConfig = optimizer_schemas.AdamWOptimizerConfig()
    scheduler: Optional[
        scheduler_schemas.LightningSchedulerConfig
    ] = scheduler_schemas.ReduceLROnPlateauLightningSchedulerConfig()


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classification_training_lightning_module_schema",
        group="tasks/lightning_module",
        node=BinaryTextClassificationTrainingLightningModuleConfig,
    )
