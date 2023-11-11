from torch import Tensor
from transformers import BatchEncoding

from cybulde.models.models import Model
from cybulde.models.transformations import Transformation
from cybulde.prediction.lightning_modules.bases import PredictionLightningModule


class BinaryTextClassificationPredictionLightningModule(PredictionLightningModule):
    def __init__(self, model: Model) -> None:
        super().__init__(model)

    def forward(self, encodings: BatchEncoding) -> Tensor:
        return self.model(encodings)

    def predict_step(self, batch: tuple[BatchEncoding, Tensor], _: int) -> tuple[Tensor, Tensor, Tensor]:
        encodings, labels = batch
        logits = self(encodings)
        input_ids = encodings["input_ids"]
        assert isinstance(input_ids, Tensor)
        return logits, input_ids, labels

    def get_transformation(self) -> Transformation:
        return self.model.get_transformation()
