from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI

from cybulde.config_schemas import data_module_schemas
from cybulde.config_schemas.base_schemas import TaskConfig
from cybulde.config_schemas.prediction import prediction_lightning_module_schemas
from cybulde.config_schemas.trainer import trainer_schemas


@dataclass
class PredictionTaskConfig(TaskConfig):
    predictions_output_dir: str = SI("${infrastructure.mlflow.artifact_uri}/predictions/${.name}")


@dataclass
class TarModelPredictionTaskConfig(PredictionTaskConfig):
    tar_model_path: str = MISSING
    lightning_module: prediction_lightning_module_schemas.PartialPredictionLightningModuleConfig = MISSING


@dataclass
class CommonPredictionTaskConfig(TarModelPredictionTaskConfig):
    _target_: str = "cybulde.prediction.tasks.common_prediction_task.CommonPredictionTask"


@dataclass
class ClassificationErrorVisualizerPredictionTaskConfig(TarModelPredictionTaskConfig):
    _target_: str = "cybulde.prediction.tasks.classification_error_visualizer_prediction_task.ClassificationErrorVisualizerPredictionTask"
    name: str = "classification_error_visualizer"
    skip_special_tokens: bool = True
    lightning_module: prediction_lightning_module_schemas.PartialPredictionLightningModuleConfig = (
        prediction_lightning_module_schemas.BinaryTextClassificationPredictionLightningModuleConfig()
    )
    trainer: trainer_schemas.TrainerConfig = trainer_schemas.PredictionTraninerConfig()


def setup_config() -> None:
    data_module_schemas.setup_config()
    prediction_lightning_module_schemas.setup_config()
    trainer_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(
        name="common_prediction_task_schema",
        group="tasks",
        node=CommonPredictionTaskConfig,
    )

    cs.store(
        name="classification_error_visualizer_schema",
        group="tasks",
        node=ClassificationErrorVisualizerPredictionTaskConfig,
    )
