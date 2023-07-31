from typing import Optional

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, BinaryF1Score
from transformers import BatchEncoding

from cybulde.models.models import Model
from cybulde.training.lightning_modules.bases import PartialOptimizerType, TrainingLightningModule
from cybulde.training.loss_functions import LossFunction
from cybulde.training.schedulers import LightningScheduler
from cybulde.utils.torch_utils import plot_confusion_matrix


class BinaryTextClassificationLightningModule(TrainingLightningModule):
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

    def forward(self, texts: BatchEncoding) -> Tensor:
        return self.model(texts)

    def training_step(self, batch: tuple[BatchEncoding, Tensor], batch_idx: int) -> Tensor:
        texts, labels = batch
        logits = self(texts)

        loss = self.loss(logits, labels)
        self.log("loss", loss, sync_dist=True)

        self.training_accuracy(logits, labels)
        self.training_f1_score(logits, labels)
        self.training_confusion_matrix(logits, labels)

        self.log("training_accuracy", self.training_accuracy, on_step=False, on_epoch=True)
        self.log("training_f1_score", self.training_f1_score, on_step=False, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs: list[Tensor]) -> None:
        confusion_matrix = self.training_confusion_matrix()
        figure = plot_confusion_matrix(confusion_matrix, ["0", "1"])
        self.experiment.log_figure(figure)  # type: ignore

    def validation_step(self, batch: tuple[BatchEncoding, Tensor], batch_idx: int) -> Tensor:
        texts, labels = batch
        logits = self(texts)

        loss = self.loss(logits, labels)
        self.log("validation_loss", loss, sync_dist=True)

        self.validation_accuracy(logits, labels)
        self.validation_f1_score(logits, labels)

        self.log("validation_accuracy", self.validation_accuracy, on_step=False, on_epoch=True)
        self.log("validation_f1_score", self.validation_f1_score, on_step=False, on_epoch=True)

        return loss

    def validation_epoch_end(self, outputs: list[Tensor]) -> None:
        confusion_matrix = self.validation_confusion_matrix()
        figure = plot_confusion_matrix(confusion_matrix, ["0", "1"])
        self.experiment.log_figure(figure)  # type: ignore
