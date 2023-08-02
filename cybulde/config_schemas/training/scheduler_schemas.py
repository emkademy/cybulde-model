from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.utils.mixins import LoggableParamsMixin


@dataclass
class SchedulerConfig(LoggableParamsMixin):
    _target_: str = MISSING
    _partial_: bool = True

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class ReduceLROnPlateauSchedulerConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "max"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-8
    verbose: bool = False


@dataclass
class LightningSchedulerConfig:
    _target_: str = MISSING
    scheduler: SchedulerConfig = MISSING
    interval: str = "epoch"
    frequency: int = 1
    monitor: str = "validation_f1_score"
    strict: bool = True
    name: Optional[str] = None


@dataclass
class CommonLightningSchedulerConfig(LightningSchedulerConfig):
    _target_: str = "cybulde.training.schedulers.CommonLightningScheduler"


@dataclass
class ReduceLROnPlateauLightningSchedulerConfig(CommonLightningSchedulerConfig):
    scheduler: SchedulerConfig = ReduceLROnPlateauSchedulerConfig(patience=5)


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="reduce_lr_on_plateau_scheduler_schema",
        group="tasks/lightning_module/scheduler",
        node=ReduceLROnPlateauLightningSchedulerConfig,
    )
