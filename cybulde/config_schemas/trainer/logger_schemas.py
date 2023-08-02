from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI


@dataclass
class LoggerConfig:
    _target_: str = MISSING


@dataclass
class MLFlowLoggerConfig(LoggerConfig):
    _target_: str = "lightning.pytorch.loggers.mlflow.MLFlowLogger"
    experiment_name: str = SI("${infrastructure.mlflow.experiment_name}")
    run_name: Optional[str] = SI("${infrastructure.mlflow.run_name}")
    tracking_uri: Optional[str] = SI("${infrastructure.mlflow.mlflow_internal_tracking_uri}")
    tags: Optional[dict[str, Any]] = None
    save_dir: Optional[str] = None
    prefix: str = ""
    artifact_location: Optional[str] = None
    run_id: Optional[str] = SI("${infrastructure.mlflow.run_id}")


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="mlflow_logger_schema", group="tasks/trainer/logger", node=MLFlowLoggerConfig)
