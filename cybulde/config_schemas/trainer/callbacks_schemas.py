from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI


@dataclass
class CallbackConfig:
    _target_: str = MISSING


@dataclass
class ModelCheckpointConfig(CallbackConfig):
    _target_: str = "lightning.pytorch.callbacks.ModelCheckpoint"
    dirpath: Optional[str] = "./data/pytorch-lightning"
    filename: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: int = 1
    mode: str = "min"
    auto_insert_metric_name: bool = False
    save_weights_only: bool = False
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None


@dataclass
class BestModelCheckpointConfig(ModelCheckpointConfig):
    dirpath: Optional[str] = SI("${infrastructure.mlflow.artifact_uri}/best-checkpoints/")
    monitor: str = MISSING
    save_last: Optional[bool] = True
    save_top_k: int = 2
    mode: str = MISSING


@dataclass
class ValidationF1ScoreBestModelCheckpointConfig(BestModelCheckpointConfig):
    monitor: str = "validation_f1_score"
    mode: str = "max"


@dataclass
class LastModelCheckpointConfig(ModelCheckpointConfig):
    dirpath: Optional[str] = SI("${infrastructure.mlflow.artifact_uri}/last-checkpoints/")
    every_n_train_steps: int = SI("${save_last_checkpoint_every_n_train_steps}")
    save_last: Optional[bool] = True
    filename: Optional[str] = "checkpoint-{epoch}"
    save_top_k: int = -1


@dataclass
class LearningRateMonitorConfig(CallbackConfig):
    _target_: str = "lightning.pytorch.callbacks.LearningRateMonitor"
    logging_interval: str = "step"


def setup_config() -> None:
    cs = ConfigStore.instance()

    cs.store(name="best_model_checkpoint_schema", group="tasks/trainer/callbacks", node=BestModelCheckpointConfig)
    cs.store(name="last_model_checkpoint_schema", group="tasks/trainer/callbacks", node=LastModelCheckpointConfig)
    cs.store(name="learning_rate_monitor_schema", group="tasks/trainer/callbacks", node=LearningRateMonitorConfig)
