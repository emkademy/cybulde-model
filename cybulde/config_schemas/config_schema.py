from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from cybulde.config_schemas import base_schemas
from cybulde.config_schemas.evaluation import evaluation_task_schemas, model_selector_schemas
from cybulde.config_schemas.infrastructure import infrastructure_schema
from cybulde.config_schemas.training import training_task_schemas


@dataclass
class Config:
    infrastructure: infrastructure_schema.InfrastructureConfig = infrastructure_schema.InfrastructureConfig()
    save_last_checkpoint_every_n_train_steps: int = 500
    seed: int = 1234
    tasks: dict[str, base_schemas.TaskConfig] = MISSING
    model_selector: Optional[model_selector_schemas.ModelSelectorConfig] = None
    registered_model_name: Optional[str] = None
    docker_image: Optional[str] = None


def setup_config() -> None:
    infrastructure_schema.setup_config()
    training_task_schemas.setup_config()
    evaluation_task_schemas.setup_config()
    model_selector_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
