from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.config_schemas import data_module_schemas
from cybulde.config_schemas.base_schemas import TaskConfig
from cybulde.config_schemas.evaluation import evaluation_lightning_module_schemas
from cybulde.config_schemas.trainer import trainer_schemas


@dataclass
class EvaluationTaskConfig(TaskConfig):
    pass


@dataclass
class TarModelEvaluationTaskConfig(EvaluationTaskConfig):
    tar_model_path: str = MISSING
    lightning_module: evaluation_lightning_module_schemas.PartialEvaluationLightningModuleConfig = MISSING


@dataclass
class CommonEvaluationTaskConfig(TarModelEvaluationTaskConfig):
    _target_: str = "cybulde.evaluation.tasks.common_evaluation_task.CommonEvaluationTask"


@dataclass
class DefaultCommonEvaluationTaskConfig(CommonEvaluationTaskConfig):
    name: str = "binary_text_evaluation_task"
    lightning_module: evaluation_lightning_module_schemas.PartialEvaluationLightningModuleConfig = (
        evaluation_lightning_module_schemas.BinaryTextEvaluationLightningModuleConfig()
    )


def setup_config() -> None:
    data_module_schemas.setup_config()
    evaluation_lightning_module_schemas.setup_config()
    trainer_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(
        name="common_evaluation_task_schema",
        group="tasks",
        node=CommonEvaluationTaskConfig,
    )
