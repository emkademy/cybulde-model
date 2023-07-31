from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Protocol, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PartialSchedulerType(Protocol):
    def __call__(
        self, optimizer: Optimizer, estimated_stepping_batches: Optional[Union[int, float]] = None
    ) -> _LRScheduler:
        ...


class LightningScheduler(ABC):
    def __init__(
        self,
        scheduler: PartialSchedulerType,
        interval: Literal["epoch", "step"] = "epoch",
        frequency: int = 1,
        monitor: str = "val_loss",
        strict: bool = True,
        name: Optional[str] = None,
    ) -> None:
        self.scheduler = scheduler
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor
        self.strict = strict
        self.name = name

    @abstractmethod
    def configure_scheduler(
        self, optimizer: Optimizer, estimated_stepping_batches: Union[int, float]
    ) -> dict[str, Any]:
        ...


class CommonLightningScheduler(LightningScheduler):
    def configure_scheduler(
        self, optimizer: Optimizer, estimated_stepping_batches: Union[int, float]
    ) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler(optimizer),
            "interval": self.interval,
            "frequency": self.frequency,
            "monitor": self.monitor,
            "strict": self.strict,
            "name": self.name,
        }
