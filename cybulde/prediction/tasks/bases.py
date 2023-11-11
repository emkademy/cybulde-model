import logging

from abc import ABC, abstractmethod
from functools import partial
from typing import Union

from lightning.pytorch.trainer import Trainer

from cybulde.config_schemas.config_schema import Config
from cybulde.config_schemas.prediction.prediction_task_schemas import PredictionTaskConfig
from cybulde.data_modules.data_modules import DataModule, PartialDataModuleType
from cybulde.models.common.exporter import TarModelLoader
from cybulde.prediction.lightning_modules.bases import PartialPredictionLightningModuleType, PredictionLightningModule


class PredictionTask(ABC):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModuleType],
        lightning_module: PredictionLightningModule,
        trainer: Trainer,
        predictions_output_dir: str,
    ) -> None:
        super().__init__()
        self.name = name
        self.trainer = trainer
        self.lightning_module = lightning_module
        self.lightning_module.eval()
        self.logger = logging.getLogger(self.__class__.__name__)

        if isinstance(data_module, DataModule):
            self.data_module = data_module
        elif isinstance(data_module, partial):
            self.data_module = data_module(model=self.lightning_module.get_model())
        self.predictions_output_dir = predictions_output_dir

    @abstractmethod
    def run(self, config: Config, task_config: PredictionTaskConfig) -> None:
        ...


class TarModelPredictionTask(PredictionTask):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModuleType],
        lightning_module: PartialPredictionLightningModuleType,
        tar_model_path: str,
        predictions_output_dir: str,
        trainer: Trainer,
    ) -> None:
        model = TarModelLoader(tar_model_path).load()
        _lightning_module = lightning_module(model=model)
        super().__init__(
            name=name,
            data_module=data_module,
            lightning_module=_lightning_module,
            predictions_output_dir=predictions_output_dir,
            trainer=trainer,
        )
