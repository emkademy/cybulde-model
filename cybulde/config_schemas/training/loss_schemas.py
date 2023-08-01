from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class LossFunctionConfig:
    _target_: str = MISSING


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
