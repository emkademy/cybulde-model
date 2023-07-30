from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from omegaconf import MISSING

from cybulde.config_schemas import transformations_schemas


@dataclass
class DataModuleConfig:
    _target_: str = MISSING
    batch_size: int = MISSING
    shuffle: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False


@dataclass
class TextClassificationDataModuleConfig(DataModuleConfig):
    _target_: str = "cybulde.data_modules.data_modules.TextClassificationDataModule"
    train_df_path: str = MISSING
    dev_df_path: str = MISSING
    test_df_path: str = MISSING
    transformation: transformations_schemas.Transformation = MISSING
    text_column_name: str = "cleaned_text"
    label_column_name: str = "label"


def setup_config() -> None:
    transformations_schemas.setup_config() 

    cs = ConfigStore.instance()
    cs.store(
        name="text_classification_data_module_schema",
        group="tasks/data_module",
        node=TextClassificationDataModuleConfig
    )
