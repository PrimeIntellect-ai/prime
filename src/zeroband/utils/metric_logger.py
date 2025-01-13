import pickle
from typing import Any, Protocol
import importlib.util


class MetricLogger(Protocol):
    def __init__(self, project, config): ...

    def log(self, metrics: dict[str, Any]): ...

    def finish(self): ...


class WandbMetricLogger(MetricLogger):
    def __init__(self, project, config, resume: bool):
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed. Please install it to use WandbMonitor.")

        import wandb

        wandb.init(
            project=project, config=config, resume="auto" if resume else None
        )  # make wandb reuse the same run id if possible

    def log(self, metrics: dict[str, Any]):
        import wandb

        wandb.log(metrics)

    def finish(self):
        import wandb

        wandb.finish()


class DummyMetricLogger(MetricLogger):
    def __init__(self, project, config, *args, **kwargs):
        self.project = project
        self.config = config
        open(self.project, "a").close()  # Create an empty file to append to

        self.data = []

    def log(self, metrics: dict[str, Any]):
        self.data.append(metrics)

    def finish(self):
        with open(self.project, "wb") as f:
            pickle.dump(self.data, f)
