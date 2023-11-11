from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cybulde.config_schemas.base_schemas import TaskConfig
from cybulde.config_schemas.config_schema import Config
from cybulde.config_schemas.evaluation import model_selector_schemas
from cybulde.config_schemas.evaluation.evaluation_task_schemas import DefaultCommonEvaluationTaskConfig
from cybulde.config_schemas.prediction.prediction_task_schemas import ClassificationErrorVisualizerPredictionTaskConfig
from cybulde.config_schemas.trainer.trainer_schemas import GPUDev
from cybulde.config_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig


@dataclass
class TestLocalBertExperiment(Config):
    tasks: dict[str, TaskConfig] = field(
        default_factory=lambda: {
            "binary_text_classification_task": DefaultCommonTrainingTaskConfig(trainer=GPUDev()),
            "binary_text_evaluation_task": DefaultCommonEvaluationTaskConfig(),
            "classification_error_visualizer": ClassificationErrorVisualizerPredictionTaskConfig(),
        }
    )
    model_selector: Optional[
        model_selector_schemas.ModelSelectorConfig
    ] = model_selector_schemas.CyberBullyingDetectionModelSelectorConfig()
    registered_model_name: Optional[str] = "bert_tiny"


FinalTestLocalBertExperiment = OmegaConf.merge(
    TestLocalBertExperiment,
    OmegaConf.from_dotlist(
        [
            "infrastructure.mlflow.experiment_name=test-run-cybulde",
            "tasks.binary_text_evaluation_task.tar_model_path=${tasks.binary_text_classification_task.tar_model_export_path}",
            "tasks.binary_text_evaluation_task.data_module=${tasks.binary_text_classification_task.data_module}",
            "tasks.binary_text_evaluation_task.trainer=${tasks.binary_text_classification_task.trainer}",
            "tasks.classification_error_visualizer.tar_model_path=${tasks.binary_text_classification_task.tar_model_export_path}",
            "tasks.classification_error_visualizer.data_module=${tasks.binary_text_classification_task.data_module}",
        ]
    ),
)

cs = ConfigStore.instance()
cs.store(name="test_local_bert", group="experiment/bert", node=FinalTestLocalBertExperiment, package="_global_")
