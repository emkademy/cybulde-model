from abc import abstractmethod
from typing import Any, Protocol

from lightning.pytorch.core import LightningModule

from cybulde.models.models import Model
from cybulde.models.transformations import Transformation


class PredictionLightningModule(LightningModule):
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        ...

    def get_model(self) -> Model:
        return self.model

    @abstractmethod
    def get_transformation(self) -> Transformation:
        ...


class PartialPredictionLightningModuleType(Protocol):
    def __call__(self, model: Model) -> PredictionLightningModule:
        ...
