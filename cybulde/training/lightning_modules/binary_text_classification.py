from collections import defaultdict
from typing import Optional

import mlflow
import torch

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, BinaryF1Score
from transformers import BatchEncoding

from cybulde.models.models import Model
from cybulde.models.transformations import Transformation
from cybulde.training.lightning_modules.bases import (
    ModelStateDictExportingTrainingLightningModule,
    PartialOptimizerType,
)
from cybulde.training.loss_functions import LossFunction
from cybulde.training.schedulers import LightningScheduler
from cybulde.utils.torch_utils import plot_confusion_matrix


class BinaryTextClassificationTrainingLightningModule(ModelStateDictExportingTrainingLightningModule):
    def __init__(
        self,
        model: Model,
        loss: LossFunction,
        optimizer: PartialOptimizerType,
        scheduler: Optional[LightningScheduler] = None,
    ) -> None:
        super().__init__(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler)

        self.training_accuracy = BinaryAccuracy()
        self.validation_accuracy = BinaryAccuracy()

        self.training_f1_score = BinaryF1Score()
        self.validation_f1_score = BinaryF1Score()

        self.training_confusion_matrix = BinaryConfusionMatrix()
        self.validation_confusion_matrix = BinaryConfusionMatrix()

        self.train_step_outputs: dict[str, list[Tensor]] = defaultdict(list)
        self.validation_step_outputs: dict[str, list[Tensor]] = defaultdict(list)

        self.pos_weight: Optional[Tensor] = None

    def set_pos_weight(self, pos_weight: Tensor) -> None:
        self.pos_weight = pos_weight

    def forward(self, texts: BatchEncoding) -> Tensor:
        output: Tensor = self.model(texts)
        return output

    def training_step(self, batch: tuple[BatchEncoding, Tensor], batch_idx: int) -> Tensor:
        texts, labels = batch
        logits = self(texts)

        self.pos_weight = self.pos_weight.to(self.device)
        loss = self.loss(logits, labels, pos_weight=self.pos_weight)
        self.log("loss", loss, sync_dist=True)

        self.training_accuracy(logits, labels)
        self.training_f1_score(logits, labels)
        self.training_confusion_matrix(logits, labels)

        self.log("training_accuracy", self.training_accuracy, on_step=False, on_epoch=True)
        self.log("training_f1_score", self.training_f1_score, on_step=False, on_epoch=True)

        self.train_step_outputs["logits"].append(logits)
        self.train_step_outputs["labels"].append(labels)

        assert isinstance(loss, Tensor)
        return loss

    def on_train_epoch_end(self) -> None:
        all_logits = torch.stack(self.train_step_outputs["logits"])
        all_labels = torch.stack(self.train_step_outputs["labels"])

        confusion_matrix = self.training_confusion_matrix(all_logits, all_labels)
        figure = plot_confusion_matrix(confusion_matrix, ["0", "1"])
        mlflow.log_figure(figure, "training_confusion_matrix.png")

        self.train_step_outputs = defaultdict(list)

    def validation_step(self, batch: tuple[BatchEncoding, Tensor], batch_idx: int) -> dict[str, Tensor]:  # type: ignore
        texts, labels = batch
        logits = self(texts)

        loss = self.loss(logits, labels)
        self.log("validation_loss", loss, sync_dist=True)

        self.validation_accuracy(logits, labels)
        self.validation_f1_score(logits, labels)

        self.log("validation_accuracy", self.validation_accuracy, on_step=False, on_epoch=True)
        self.log("validation_f1_score", self.validation_f1_score, on_step=False, on_epoch=True)

        self.validation_step_outputs["logits"].append(logits)
        self.validation_step_outputs["labels"].append(labels)

        return {"loss": loss, "predictions": logits, "labels": labels}

    def on_validation_epoch_end(self) -> None:
        all_logits = torch.stack(self.validation_step_outputs["logits"])
        all_labels = torch.stack(self.validation_step_outputs["labels"])

        confusion_matrix = self.validation_confusion_matrix(all_logits, all_labels)
        figure = plot_confusion_matrix(confusion_matrix, ["0", "1"])
        mlflow.log_figure(figure, "validation_confusion_matrix.png")

        self.validation_step_outputs = defaultdict(list)

    def get_transformation(self) -> Transformation:
        return self.model.get_transformation()

    def export_model_state_dict(self, checkpoint_path: str) -> str:
        return self.common_export_model_state_dict(checkpoint_path)
