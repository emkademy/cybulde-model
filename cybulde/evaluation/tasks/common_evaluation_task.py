from typing import TYPE_CHECKING, Union

from lightning.pytorch import Trainer

from cybulde.data_modules.data_modules import DataModule, PartialDataModule
from cybulde.evaluation.lightning_modules.bases import PartialEvaluationLightningModuleType
from cybulde.evaluation.tasks.bases import TarModelEvaluationTask
from cybulde.utils.mlflow_utils import activate_mlflow

if TYPE_CHECKING:
    from cybulde.config_schemas.config_schema import Config
    from cybulde.config_schemas.evaluation.evaluation_task_schemas import EvaluationTaskConfig


class CommonEvaluationTask(TarModelEvaluationTask):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModule],
        lightning_module: PartialEvaluationLightningModuleType,
        trainer: Trainer,
        tar_model_path: str,
    ) -> None:
        super().__init__(
            name=name,
            data_module=data_module,
            lightning_module=lightning_module,
            trainer=trainer,
            tar_model_path=tar_model_path,
        )

    def run(self, config: "Config", task_config: "EvaluationTaskConfig") -> None:
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name

        with activate_mlflow(experiment_name=experiment_name, run_id=run_id, run_name=run_name) as _:
            self.trainer.test(model=self.lightning_module, datamodule=self.data_module)
