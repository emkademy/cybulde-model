import os

from typing import Union

import pandas as pd
import torch

from lightning.pytorch import Trainer
from torch import Tensor

from cybulde.config_schemas.config_schema import Config
from cybulde.config_schemas.prediction.prediction_task_schemas import TarModelPredictionTaskConfig
from cybulde.data_modules.data_modules import DataModule
from cybulde.prediction.tasks.bases import (
    PartialDataModuleType,
    PartialPredictionLightningModuleType,
    TarModelPredictionTask,
)
from cybulde.utils.io_utils import list_paths, open_file, remove_path
from cybulde.utils.torch_utils import global_rank_zero_first


class ClassificationErrorVisualizerPredictionTask(TarModelPredictionTask):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModuleType],
        lightning_module: PartialPredictionLightningModuleType,
        trainer: Trainer,
        tar_model_path: str,
        predictions_output_dir: str,
        skip_special_tokens: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            data_module=data_module,
            lightning_module=lightning_module,
            tar_model_path=tar_model_path,
            predictions_output_dir=predictions_output_dir,
            trainer=trainer,
        )
        self.transformation = self.lightning_module.get_transformation()
        self.skip_special_tokens = skip_special_tokens

    def run(self, config: Config, task_config: TarModelPredictionTaskConfig) -> None:
        self.trainer.predict(model=self.lightning_module, datamodule=self.data_module, return_predictions=False)
        torch.distributed.barrier()

        with global_rank_zero_first():
            if self.trainer.is_global_zero:
                all_logits = []
                all_input_ids = []
                all_labels = []
                pt_file_paths = list_paths(self.predictions_output_dir, check_path_suffix=True, path_suffix=".pt")
                for pt_file_path in pt_file_paths:
                    with open_file(pt_file_path, "rb") as f:
                        outputs = torch.load(f)

                    _logits, input_ids, _labels = list(zip(*outputs))
                    logits = torch.cat(_logits)
                    labels = torch.cat(_labels)

                    all_logits.append(logits)
                    all_labels.append(labels)
                    all_input_ids.extend(input_ids)

                all_logits = torch.cat(all_logits).view(-1)
                all_labels = torch.cat(all_labels).view(-1)
                all_input_ids = [single_input_ids for input_ids in all_input_ids for single_input_ids in input_ids]

                predicted_labels = all_logits >= 0.5

                error_mask = predicted_labels != all_labels

                error_texts = self.get_error_texts(all_input_ids, error_mask)
                error_labels = all_labels[error_mask].numpy().tolist()

                error_df = pd.DataFrame(
                    {
                        "text": error_texts,
                        "gt_label": error_labels,
                    }
                )

                error_df_save_path = os.path.join(self.predictions_output_dir, "errors.csv")
                error_df.to_csv(error_df_save_path, index=False)

                for pt_file_path in pt_file_paths:
                    remove_path(pt_file_path)

    def get_error_texts(self, all_input_ids: list[Tensor], error_mask: Tensor) -> list[str]:
        error_indices = error_mask.nonzero()
        all_error_texts = []
        for error_index in error_indices:
            input_ids = all_input_ids[error_index].view(1, -1)
            error_text = self.transformation.decode(input_ids, skip_special_tokens=self.skip_special_tokens)[0]
            all_error_texts.append(error_text)
        return all_error_texts
