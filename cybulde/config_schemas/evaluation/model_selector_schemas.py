from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI


@dataclass
class MetricComparerConfig:
    _target_: str = "cybulde.evaluation.model_selector.MetricComparer"
    bigger_is_better: bool = MISSING
    can_be_equal: bool = False
    metric_name: str = MISSING
    threshold: float = 0.0


@dataclass
class BinaryF1ScoreMetricComparerConfig(MetricComparerConfig):
    bigger_is_better: bool = True
    metric_name: str = "test_f1_score"


@dataclass
class ModelSizeMetricComparerConfig(MetricComparerConfig):
    bigger_is_better: bool = False
    metric_name: str = "model_size"
    can_be_equal: bool = True


@dataclass
class ModelSelectorConfig:
    _target_: str = "cybulde.evaluation.model_selector.ModelSelector"
    mlflow_run_id: Optional[str] = SI("${infrastructure.mlflow.run_id}")
    must_be_better_metric_comparers: dict[str, MetricComparerConfig] = field(default_factory=lambda: {})
    to_be_thresholded_metric_comparers: dict[str, MetricComparerConfig] = field(default_factory=lambda: {})
    threshold: float = 0.0


@dataclass
class CyberBullyingDetectionModelSelectorConfig(ModelSelectorConfig):
    must_be_better_metric_comparers: dict[str, MetricComparerConfig] = field(
        default_factory=lambda: {
            "f1_score": BinaryF1ScoreMetricComparerConfig(),
            "model_size": ModelSizeMetricComparerConfig(),
        }
    )


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="metric_comparer_schema", group="model_selector/metric_comparers", node=MetricComparerConfig)
    cs.store(name="model_selector_schema", group="model_selector", node=ModelSelectorConfig)
