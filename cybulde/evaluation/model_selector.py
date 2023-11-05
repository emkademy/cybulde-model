from typing import Any, Optional

from mlflow.entities import Run

from cybulde.utils.mlflow_utils import get_best_run, get_client
from cybulde.utils.utils import get_logger


class MetricComparer:
    def __init__(self, bigger_is_better: bool, can_be_equal: bool, metric_name: str, threshold: float = 0.0) -> None:
        self.bigger_is_better = bigger_is_better
        self.can_be_equal = can_be_equal
        self.metric_name = metric_name
        self.threshold = threshold

    def is_metric_better(self, run: Run, best_run_data: dict[str, Any]) -> bool:
        if not best_run_data:
            return True

        current_metric_value = self.get_current_metric_value(run)
        best_metric_value = best_run_data[f"metrics.{self.metric_name}"]

        if self.can_be_equal and current_metric_value == best_metric_value:
            return True

        if self.bigger_is_better:
            current_metric_value -= self.threshold
            result = current_metric_value > best_metric_value
            assert isinstance(result, bool)
            return result
        else:
            current_metric_value += self.threshold
            result = current_metric_value < best_metric_value
            assert isinstance(result, bool)
            return result

    def get_current_metric_value(self, run: Run) -> float:
        current_metric_value = run.data.metrics.get(self.metric_name, None)
        if current_metric_value is None:
            raise RuntimeError(f"Metric: {self.metric_name} couldn't be found on MLFlow. Was it logged?")
        assert isinstance(current_metric_value, float)
        return current_metric_value


class ModelSelector:
    def __init__(
        self,
        mlflow_run_id: str,
        must_be_better_metric_comparers: dict[str, MetricComparer] = {},
        to_be_thresholded_metric_comparers: dict[str, MetricComparer] = {},
        threshold: float = 0.0,
    ) -> None:
        if not must_be_better_metric_comparers and not to_be_thresholded_metric_comparers:
            raise ValueError(
                "Both 'must_be_better_metric_comparers' and 'to_be_thresholded_metric_comparers' cannot be empty..."
            )

        self.logger = get_logger(self.__class__.__name__)

        self.mlflow_run_id = mlflow_run_id
        self.must_be_better_metric_comparers = must_be_better_metric_comparers
        self.to_be_thresholded_metric_comparers = to_be_thresholded_metric_comparers
        self.threshold = threshold

        client = get_client()
        self.run = client.get_run(mlflow_run_id)
        self.best_run_data = get_best_run()
        self.new_best_run_tag: Optional[str] = None

    def is_selected(self) -> bool:
        is_selected = self._is_selected(self.run)
        if is_selected:
            self.new_best_run_tag = self.get_new_best_run_tag()
        return is_selected

    def _is_selected(self, run: Run) -> bool:
        for metric_name, metric_comparer in self.must_be_better_metric_comparers.items():
            if not metric_comparer.is_metric_better(run, self.best_run_data):
                self.logger.info(f"'{metric_name}' is a must have metric, and its value is not better than before...")
                return False

        hits = []
        for metric_comparer in self.to_be_thresholded_metric_comparers.values():
            is_metric_better = metric_comparer.is_metric_better(run, self.best_run_data)
            hits.append(int(is_metric_better))

        if not hits:
            return True

        mean_hits = sum(hits) / len(hits)
        return mean_hits > self.threshold

    def get_new_best_run_tag(self) -> str:
        if len(self.best_run_data) == 0:
            return "v1"
        last_tag: str = self.best_run_data["tags.best_run"]
        last_version = int(last_tag[1:])
        return f"v{last_version + 1}"
