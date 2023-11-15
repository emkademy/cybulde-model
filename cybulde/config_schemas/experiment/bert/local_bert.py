from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cybulde.config_schemas.base_schemas import TaskConfig
from cybulde.config_schemas.config_schema import Config
from cybulde.config_schemas.evaluation import model_selector_schemas
from cybulde.config_schemas.evaluation.evaluation_task_schemas import DefaultCommonEvaluationTaskConfig
from cybulde.config_schemas.trainer.trainer_schemas import GPUProd
from cybulde.config_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig


@dataclass
class LocalBertExperiment(Config):
    tasks: dict[str, TaskConfig] = field(
        default_factory=lambda: {
            "binary_text_classification_task": DefaultCommonTrainingTaskConfig(trainer=GPUProd()),
            "binary_text_evaluation_task": DefaultCommonEvaluationTaskConfig(),
        }
    )
    model_selector: Optional[
        model_selector_schemas.ModelSelectorConfig
    ] = model_selector_schemas.CyberBullyingDetectionModelSelectorConfig()
    registered_model_name: Optional[str] = "bert_tiny"


FinalLocalBertExperiment = OmegaConf.merge(
    LocalBertExperiment,
    OmegaConf.from_dotlist(
        [
            "infrastructure.mlflow.experiment_name=cybulde",
            "tasks.binary_text_classification_task.data_module.batch_size=1024",
            "tasks.binary_text_evaluation_task.tar_model_path=${tasks.binary_text_classification_task.tar_model_export_path}",
            "tasks.binary_text_evaluation_task.data_module=${tasks.binary_text_classification_task.data_module}",
            "tasks.binary_text_evaluation_task.trainer=${tasks.binary_text_classification_task.trainer}",
        ]
    ),
)

cs = ConfigStore.instance()
cs.store(name="local_bert", group="experiment/bert", node=FinalLocalBertExperiment, package="_global_")
