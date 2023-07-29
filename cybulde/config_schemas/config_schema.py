from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass

from cybulde.config_schemas.infrastructure.infrastructure_schema import InfrastructureConfig


@dataclass
class Config:
    infrastructure: InfrastructureConfig = InfrastructureConfig()


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
