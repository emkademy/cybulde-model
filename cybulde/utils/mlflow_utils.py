import dataclasses
import os

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Iterable, Optional

import mlflow

from cybulde.utils.mixins import LoggableParamsMixin

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if TYPE_CHECKING:
    from cybulde.config_schemas.config_schema import Config


@contextmanager  # type: ignore
def activate_mlflow(
    experiment_name: Optional[str] = None,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Iterable[mlflow.ActiveRun]:
    set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:  # type: ignore
        yield run


def set_experiment(experiment_name: Optional[str] = None) -> None:
    if experiment_name is None:
        experiment_name = "Default"

    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.RestException:
        pass

    mlflow.set_experiment(experiment_name)


def log_artifacts_for_reproducibility() -> None:
    locations_to_store = ["./cybulde", "./docker", "./pyproject.toml", "./poetry.lock"]

    for location_to_store in locations_to_store:
        mlflow.log_artifact(location_to_store, "reproduction")


def log_training_hparams(config: "Config") -> None:
    logged_nodes = set()

    def loggable_params(node: Any, path: list[str]) -> Generator[tuple[str, Any], None, None]:
        if isinstance(node, LoggableParamsMixin) and id(node) not in logged_nodes:
            for param_name in node.loggable_params():
                yield ".".join(path + [param_name]), getattr(node, param_name)
            logged_nodes.add(id(node))
        children = None
        if isinstance(node, dict):
            children = node.items()
        if dataclasses.is_dataclass(node):
            children = ((f.name, getattr(node, f.name)) for f in dataclasses.fields(node))  # type: ignore

        if children is None:
            return
        for key, val in children:
            for item in loggable_params(val, path + [key]):
                yield item

    params = dict(loggable_params(config, []))
    mlflow.log_params(params)
