from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from omegaconf import DictConfig

from cybulde.utils.config_utils import get_config_and_dict_config, save_config_as_yaml
from cybulde.utils.mlflow_utils import activate_mlflow, log_artifacts_for_reproducibility, log_training_hparams

if TYPE_CHECKING:
    from cybulde.config_schemas.config_schema import Config


@get_config_and_dict_config(config_path="../configs", config_name="config")  # type: ignore
def generate_final_config(config: "Config", dict_config: DictConfig) -> None:
    run: mlflow.ActiveRun
    with activate_mlflow(
        config.infrastructure.mlflow.experiment_name,
        run_id=config.infrastructure.mlflow.run_id,
        run_name=config.infrastructure.mlflow.run_name,
    ) as run:
        run_id: str = run.info.run_id
        experiment_id: str = run.info.experiment_id
        artifact_uri: str = run.info.artifact_uri

        dict_config.infrastructure.mlflow.artifact_uri = artifact_uri
        dict_config.infrastructure.mlflow.run_id = run_id
        dict_config.infrastructure.mlflow.experiment_id = experiment_id

        config_save_dir = Path("./cybulde/configs/automatically_generated/")
        config_save_dir.mkdir(parents=True, exist_ok=True)
        (config_save_dir / "__init__.py").touch(exist_ok=True)

        yaml_config_save_path = config_save_dir / "config.yaml"
        save_config_as_yaml(dict_config, str(yaml_config_save_path))
        mlflow.log_artifact(str(yaml_config_save_path))

        log_training_hparams(config)
        log_artifacts_for_reproducibility()


if __name__ == "__main__":
    generate_final_config()
