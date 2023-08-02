from abc import abstractmethod
from typing import Any, Protocol

from lightning.pytorch import LightningModule
from torch import Tensor

from cybulde.models.models import Model
from cybulde.models.transformations import Transformation


class EvaluationLightningModule(LightningModule):
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        ...

    @abstractmethod
    def get_transformation(self) -> Transformation:
        ...


class PartialEvaluationLightningModuleType(Protocol):
    def __call__(self, model: Model) -> EvaluationLightningModule:
        ...
