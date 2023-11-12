from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from lightning.pytorch import Trainer

from cybulde.data_modules.data_modules import DataModule, PartialDataModuleType
from cybulde.evaluation.lightning_modules.bases import EvaluationLightningModule, PartialEvaluationLightningModuleType
from cybulde.models.common.exporter import TarModelLoader

if TYPE_CHECKING:
    from cybulde.config_schemas.config_schema import Config
    from cybulde.config_schemas.evaluation.evaluation_task_schemas import EvaluationTaskConfig


class EvaluationTask(ABC):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModuleType],
        lightning_module: EvaluationLightningModule,
        trainer: Trainer,
    ) -> None:
        super().__init__()

        self.name = name
        self.trainer = trainer
        self.lightning_module = lightning_module
        self.lightning_module.eval()

        if isinstance(data_module, DataModule):
            self.data_module = data_module
        else:
            self.data_module = data_module(transformation=self.lightning_module.get_transformation())

    @abstractmethod
    def run(self, config: "Config", task_config: "EvaluationTaskConfig") -> None:
        ...


class TarModelEvaluationTask(EvaluationTask):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModuleType],
        lightning_module: PartialEvaluationLightningModuleType,
        trainer: Trainer,
        tar_model_path: str,
    ) -> None:
        model = TarModelLoader(tar_model_path).load()
        _lightning_module = lightning_module(model=model)
        super().__init__(name=name, data_module=data_module, lightning_module=_lightning_module, trainer=trainer)
