from lightning.pytorch.trainer import Trainer

from cybulde.config_schemas.config_schema import Config
from cybulde.config_schemas.prediction.task_schema import TarModelPredictionTaskConfig
from cybulde.prediction.tasks.bases import (
    PartialDataModuleType,
    PartialPredictionLightningModuleType,
    TarModelPredictionTask,
)


class CommonPredictionTask(TarModelPredictionTask):
    def __init__(
        self,
        name: str,
        data_module: PartialDataModuleType,
        lightning_module: PartialPredictionLightningModuleType,
        trainer: Trainer,
        tar_model_path: str,
        predictions_output_dir: str,
    ) -> None:
        super().__init__(
            name=name,
            data_module=data_module,
            lightning_module=lightning_module,
            tar_model_path=tar_model_path,
            predictions_output_dir=predictions_output_dir,
            trainer=trainer,
        )

    def run(self, config: Config, task_config: TarModelPredictionTaskConfig) -> None:
        self.trainer.predict(model=self.lightning_module, datamodule=self.data_module, return_predictions=False)
