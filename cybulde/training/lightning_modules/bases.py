import os

from abc import abstractmethod
from typing import Any, Callable, Iterable, Optional, Union

import mlflow
import torch

from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import Optimizer

from cybulde.models.models import Model
from cybulde.models.transformations import Transformation
from cybulde.training.loss_functions import LossFunction
from cybulde.training.schedulers import LightningScheduler
from cybulde.utils.io_utils import open_file
from cybulde.utils.utils import get_logger

PartialOptimizerType = Callable[[Union[Iterable[Tensor], dict[str, Iterable[Tensor]]]], Optimizer]


class TrainingLightningModule(LightningModule):
    def __init__(
        self,
        model: Model,
        loss: LossFunction,
        optimizer: PartialOptimizerType,
        scheduler: Optional[LightningScheduler] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
        self.partial_optimizer = optimizer
        self.scheduler = scheduler

        self.model_size = self._calculate_model_size()
        self.logging_logger = get_logger(self.__class__.__name__)

    def _calculate_model_size(self) -> float:
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() & param.element_size()

        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def configure_optimizers(self) -> Union[Optimizer, tuple[list[Optimizer], list[dict[str, Any]]]]:
        optimizer = self.partial_optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler.configure_scheduler(
                optimizer=optimizer, estimated_stepping_batches=self.trainer.estimated_stepping_batches
            )
            return [optimizer], [scheduler]

        return optimizer

    def on_train_end(self) -> None:
        try:
            mlflow.log_metric("model_size", self.model_size)
        except Exception:
            pass
        return super().on_train_end()

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ...

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ...

    @abstractmethod
    def get_transformation(self) -> Transformation:
        ...


class ModelStateDictExportingTrainingLightningModule(TrainingLightningModule):
    @abstractmethod
    def export_model_state_dict(self, checkpoint_path: str) -> str:
        """
        Export model state dict from LightningModule checkpoint and save it
        to the same location as the checkpoint_path, and return the save path
        """

    def common_export_model_state_dict(self, checkpoint_path: str) -> str:
        with open_file(checkpoint_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))["state_dict"]

        model_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("loss."):
                model_state_dict[key.replace("model.", "", 1)] = value

        model_state_dict_save_path = os.path.join(os.path.dirname(checkpoint_path), "model_state_dict.pth")

        with open_file(model_state_dict_save_path, "wb") as f:
            torch.save(model_state_dict, f)

        return model_state_dict_save_path
