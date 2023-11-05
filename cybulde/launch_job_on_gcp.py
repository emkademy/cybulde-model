from typing import TYPE_CHECKING

import mlflow

from hydra.utils import instantiate

from cybulde.utils.config_utils import get_config
from cybulde.utils.gcp_utils import TrainingInfo

if TYPE_CHECKING:
    from cybulde.config_schemas.config_schema import Config


@get_config(
    config_path="../configs/automatically_generated", config_name="config", to_object=False, return_dict_config=True
)
def run(config: "Config") -> None:
    run_id = config.infrastructure.mlflow.run_id
    assert run_id is not None

    instance_group_creator = instantiate(config.infrastructure.instance_group_creator)
    instance_ids = instance_group_creator.launch_instance_group()
    training_info = TrainingInfo(
        project_id=config.infrastructure.project_id,
        zone=config.infrastructure.zone,
        instance_group_name=config.infrastructure.instance_group_creator.name,
        instance_ids=instance_ids,
        mlflow_experiment_url=config.infrastructure.mlflow.experiment_url,
    )
    mlflow.start_run(run_id=run_id, description=training_info.get_job_info_message())
    training_info.print_job_info()


if __name__ == "__main__":
    run()
