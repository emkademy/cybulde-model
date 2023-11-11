import os

from typing import Any

import torch

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from torch import Tensor

from cybulde.utils.io_utils import make_dirs, open_file, rename_file


class CustomPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str) -> None:
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir
        make_dirs(self.output_dir)

    def write_on_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, predictions: Any, batch_indices: Tensor
    ) -> None:
        temp_save_path = os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt.temp")
        save_path = os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt")
        with open_file(temp_save_path, "wb") as f:
            torch.save(predictions, f)

        rename_file(temp_save_path, save_path)
