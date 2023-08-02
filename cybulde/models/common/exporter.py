import os
import sys
import tarfile
import tempfile

from typing import Any

import torch

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cybulde.models.common.io_utils import cache_gcs_resource_locally, copy_file
from cybulde.models.common.utils import get_global_rank, get_local_rank, global_rank_zero_first, local_rank_zero_first
from cybulde.utils.utils import get_logger

MODELS_MODULE_PATH = "cybulde/models"
TEMP_MODELS_MODULE_PATH = "temp_module/models"
MODEL_CONFIG_PATH = "model_config.yaml"
STATE_DICT_PATH = "model_state_dict.pth"
EXPORTED_MODEL_FILE_NAME = "exported_model.tar.gz"


class TarModelExporter:
    def __init__(
        self,
        model_state_dict_path: str,
        model_config: Any,
        tar_model_export_path: str,
    ) -> None:
        self.model_state_dict_path = model_state_dict_path
        self.model_config = model_config
        self.tar_model_export_path = tar_model_export_path

        self.logger = get_logger(self.__class__.__name__)

    def export(self) -> None:
        with global_rank_zero_first():
            if get_global_rank() in [0, -1]:
                state_dict_path = self.download_model_state_dict()
                model_config_path = self.save_model_config()

                local_tar_path = os.path.join(tempfile.gettempdir(), EXPORTED_MODEL_FILE_NAME)
                with tarfile.open(local_tar_path, "w:gz") as tar:
                    tar.add(MODELS_MODULE_PATH, arcname=TEMP_MODELS_MODULE_PATH)
                    tar.add(state_dict_path, arcname=STATE_DICT_PATH)
                    tar.add(model_config_path, arcname=MODEL_CONFIG_PATH)

                copy_file(local_tar_path, self.tar_model_export_path)

                self.logger.info("Model exported successfully!")

    def download_model_state_dict(self) -> str:
        return cache_gcs_resource_locally(self.model_state_dict_path)

    def save_model_config(self) -> str:
        model_config_save_path = os.path.join(tempfile.gettempdir(), MODEL_CONFIG_PATH)
        OmegaConf.save(self.model_config, model_config_save_path)
        return model_config_save_path


class TarModelLoader:
    def __init__(self, exported_model_path: str) -> None:
        self.exported_model_path = exported_model_path
        self.replace_module_from = MODELS_MODULE_PATH.split("/")[0]
        self.replace_module_to = TEMP_MODELS_MODULE_PATH.split("/")[0]
        self.logger = get_logger(self.__class__.__name__)

    def load(self) -> Any:
        temp_export_path = "/tmp/temp_cybulde"

        with local_rank_zero_first():
            if get_local_rank() in [0, -1]:
                self.extract_tar_gz(temp_export_path)

        model_config = self.load_model_config(temp_export_path)
        model = self.load_model(temp_export_path, model_config)
        return model

    def extract_tar_gz(self, export_path: str) -> None:
        local_exported_model_path = cache_gcs_resource_locally(self.exported_model_path)
        with tarfile.open(local_exported_model_path, "r:gz") as tar:
            tar.extractall(path=export_path)

    def load_model_config(self, model_dir: str) -> Any:
        model_config = OmegaConf.load(f"{model_dir}/{MODEL_CONFIG_PATH}")
        model_config = self._replace_module_in_model_config(model_config)
        return model_config

    def load_model(self, model_dir: str, model_config: Any) -> Any:
        sys.path.append(model_dir)

        model = instantiate(model_config)
        state_dict = torch.load(f"{model_dir}/{STATE_DICT_PATH}")
        model.load_state_dict(state_dict)

        sys.path.remove(model_dir)
        return model

    def _replace_module_in_model_config(self, config: Any) -> Any:
        for key, value in config.items():
            if isinstance(value, (dict, DictConfig)):
                self._replace_module_in_model_config(value)

            if key == "_target_":
                assert isinstance(value, str)
                config[key] = value.replace(self.replace_module_from, self.replace_module_to)

        return config
