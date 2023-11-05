from collections import defaultdict

import mlflow
import torch

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, BinaryF1Score
from transformers import BatchEncoding

from cybulde.evaluation.lightning_modules.bases import EvaluationLightningModule
from cybulde.models.models import Model
from cybulde.models.transformations import Transformation
from cybulde.utils.torch_utils import plot_confusion_matrix


class BinaryTextEvaluationLightningModule(EvaluationLightningModule):
    def __init__(
        self,
        model: Model,
    ) -> None:
        super().__init__(model=model)

        self.test_accuracy = BinaryAccuracy()
        self.test_f1_score = BinaryF1Score()
        self.test_confusion_matrix = BinaryConfusionMatrix()

        self.test_step_outputs: dict[str, list[Tensor]] = defaultdict(list)

    def forward(self, texts: BatchEncoding) -> Tensor:
        output: Tensor = self.model(texts)
        return output

    def test_step(self, batch: tuple[BatchEncoding, Tensor], batch_idx: int) -> None:  # type: ignore
        texts, labels = batch
        logits = self(texts)

        self.test_accuracy(logits, labels)
        self.test_f1_score(logits, labels)
        self.test_confusion_matrix(logits, labels)

        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test_f1_score", self.test_f1_score, on_step=False, on_epoch=True)

        self.test_step_outputs["logits"].append(logits)
        self.test_step_outputs["labels"].append(labels)

    def on_test_epoch_end(self) -> None:
        all_logits = torch.stack(self.test_step_outputs["logits"])
        all_labels = torch.stack(self.test_step_outputs["labels"])

        confusion_matrix = self.test_confusion_matrix(all_logits, all_labels)
        figure = plot_confusion_matrix(confusion_matrix, ["0", "1"])
        mlflow.log_figure(figure, "test_confusion_matrix.png")

        self.test_step_outputs = defaultdict(list)

    def get_transformation(self) -> Transformation:
        return self.model.get_transformation()
