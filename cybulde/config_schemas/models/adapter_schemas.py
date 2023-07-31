from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class AdapterConfig:
    _target_: str = MISSING


@dataclass
class MLPWithPoolingConfig(AdapterConfig):
    _target_: str = "cybulde.models.adapters.MLPWithPooling"
    output_feature_sizes: list[int] = MISSING
    biases: Optional[list[bool]] = None
    activation_fns: Optional[list[Optional[str]]] = None
    dropout_drop_probs: Optional[list[float]] = None
    batch_norms: Optional[list[bool]] = None
    order: str = "LABDN"
    standardize_input: bool = True
    pooling_method: Optional[str] = None
    output_attribute_to_use: Optional[str] = None


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="mlp_with_pooling_schema",
        group="tasks/lightning_module/model/adapter",
        node=MLPWithPoolingConfig,
    )
