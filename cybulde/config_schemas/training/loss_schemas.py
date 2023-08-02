from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.utils.mixins import LoggableParamsMixin


@dataclass
class LossFunctionConfig(LoggableParamsMixin):
    _target_: str = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class BCEWithLogitsLossConfig(LossFunctionConfig):
    _target_: str = "cybulde.training.loss_functions.BCEWithLogitsLoss"
    reduction: str = "mean"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="bce_with_logits_loss_schema",
        group="tasks/lightning_module/loss",
        node=BCEWithLogitsLossConfig,
    )
