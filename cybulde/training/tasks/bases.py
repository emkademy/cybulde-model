from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Union

from lightning.pytorch import Trainer

from cybulde.data_modules.data_modules import DataModule, PartialDataModule
from cybulde.training.lightning_modules.bases import TrainingLightningModule
from cybulde.utils.utils import get_logger

if TYPE_CHECKING:
    from cybulde.config_schemas.config_schema import Config
    from cybulde.config_schemas.training.training_task_schemas import TrainingTaskConfig


class TrainingTask(ABC):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModule],
        lightning_module: TrainingLightningModule,
        trainer: Trainer,
        best_training_checkpoint: str,
        last_training_checkpoing: str,
    ) -> None:
        super().__init__()
        self.name = name
        self.trainer = trainer
        self.best_training_checkpoint = best_training_checkpoint
        self.last_training_checkpoing = last_training_checkpoing
        self.logger = get_logger(self.__class__.__name__)

        self.lightning_module = lightning_module

        if isinstance(data_module, partial):
            transformation = self.lightning_module.get_transformation()
            self.data_module = data_module(transformation=transformation)
        else:
            self.data_module = data_module

    @abstractmethod
    def run(self, config: "Config", task_config: "TrainingTaskConfig") -> None:
        ...
