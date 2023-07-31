from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class HeadConfig:
    _target_: str = MISSING


@dataclass
class SigmoidHeadConfig(HeadConfig):
    _target_: str = "cybulde.models.heads.SigmoidHead"
    in_features: int = MISSING
    out_features: int = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="sigmoid_head_schema",
        group="tasks/lightning_module/model/head",
        node=SigmoidHeadConfig,
    )
