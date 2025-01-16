from typing import Any, Literal, TypeAlias
import os

from pydantic import model_validator
from pydantic_config import BaseConfig

from zeroband.collectives import Compression
from zeroband.models.llama.model import AttnFnType


class DataConfig(BaseConfig):
    dataset_name_or_paths: str = "datasets/fineweb-edu"
    val_dataset_name_or_paths: str | None = None
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 4
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    dataset_ratio: str | None = None
    data_rank: int | None = None
    data_world_size: int | None = None
    reverse_data_files: bool = False
    split_by_data_rank: bool = True


class AdamConfig(BaseConfig):
    type: Literal["adam"] = (
        "adam"  # the literal is used to distinguish between the different optimizers configuration in the union type
    )
    lr: float = 4e-4
    weight_decay: float = 0.1
    betas1: float = 0.9
    betas2: float = 0.95


class SoapConfig(BaseConfig):
    type: Literal["soap"] = "soap"
    lr: float = 4e-4
    weight_decay: float = 1e-05
    betas1: float = 0.9
    betas2: float = 0.95

    max_preconditioner_dim: int = 8192
    precondition_frequency: int = 100


OptimizersConfig: TypeAlias = AdamConfig | SoapConfig


class OptimConfig(BaseConfig):
    optim: OptimizersConfig = AdamConfig()

    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512

    z_loss: bool = False
    z_loss_weight: float = 2e-4
    num_chunks: int | None = None


class DilocoConfig(BaseConfig):
    outer_lr: float = 0.7
    inner_steps: int
    compression: Compression = Compression.NO

    retry_all_reduce: int = 3


class MemoryProfilerConfig(BaseConfig):
    freq: int = 10
    snapshot_dir: str


class TrainConfig(BaseConfig):
    micro_bs: int
    torch_compile: bool = True
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard

    reduce_fp32: bool = False  # should be True if SXM. Keep to false as default for backward compatibility

    log_model_hash: bool = False

    memory_profiler: MemoryProfilerConfig | None = None

    torch_profiler: bool = False

    sequence_packing: bool = True

    attn_fn: AttnFnType = "flex"


class MonitorConfig(BaseConfig):
    log_flush_interval: int = 10
    base_url: str | None = None
    auth_token: str | None = None


class RemoteConfig(BaseConfig):
    path: str  # could be a s3 path
    interval: int


class CkptConfig(BaseConfig):
    path: str | None = None
    interval: int | None = None
    topk: int | None = None

    remote: RemoteConfig | None = None

    remote_data_path: str | None = None
    remote_data_load: bool = False

    resume: str | None = None

    skip_dataloader: bool = False

    live_recovery_rank_src: int | None = None

    data_path: str | None = None

    token_count: int | None = None

    @model_validator(mode="after")
    def validate_path_and_interval(self):
        if (self.path is None) != (self.interval is None):
            raise ValueError("path and interval must be both set or both None")
        if self.path is None and self.remote is not None:
            raise ValueError("remote_path is set but path is not set")

        return self

    @model_validator(mode="after")
    def validate_remote_data_path(self):
        if self.remote_data_load and self.data_path is not None:
            raise ValueError("remote_data_load and data_path are mutually exclusive")

        if self.remote_data_load and self.remote_data_path is None:
            raise ValueError("remote_data_load is set but remote_data_path is not set")
        return self


class Config(BaseConfig):
    # main config
    name_model: Literal["debugmodel", "150M", "271M", "1B", "7B", "10B", "13B", "26B", "70B"] = "150M"
    type_model: Literal["llama2", "llama3"] = "llama3"

    # Project/Run
    project: str = "zeroband"
    run_id: str | None = None
    run_name: str | None = None

    # Logger
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"
    wandb_resume: bool = False
    log_level: Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_all_rank: bool = False

    # sub config
    diloco: DilocoConfig | None = None
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig
    monitor: MonitorConfig | None = None

    ckpt: CkptConfig = CkptConfig()

    @model_validator(mode="after")
    def ckpt_diloco_step(self):
        if self.ckpt is not None and self.ckpt.interval is not None and self.diloco is not None:
            assert (
                self.ckpt.interval % self.diloco.inner_steps == 0
            ), "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"
        return self

    @model_validator(mode="after")
    def validate_live_recovery_rank_src(self):
        if self.ckpt is not None and self.ckpt.live_recovery_rank_src is not None and self.diloco is None:
            raise ValueError("live_recovery_rank_src is only supported with diloco")
        return self


def get_env_config(config: Config | None, item: str | None, default: Any | None = None) -> Any:
    """
    Get a config value from the environment or the config.
        item: item is of the form "train.memory_profiler.freq"
        default: default value if not found

    If either config or item are None, returns default. This is so you can call get_logger() as before.

    Examples:
    ```
    # Returns ZERO_BAND_RUN_NAME if set in env.
    # Otherwise returns config.run_name.
    get_env_config(config, "run_name")
    ```
    ```
    # Returns ZERO_BAND_TRAIN_MEMORY_PROFILER_FREQ if set in env.
    # Then returns 10 if train or config.train.memory_profiler are None.
    # Otherwise, returns the value of config.train.memory_profiler.freq.
    get_env_config(config, "train.memory_profiler.freq", 10)
    ```

    """

    if config is None or item is None:
        return default

    # Check env
    env_name = "ZERO_BAND_" + item.upper().replace(".", "_")
    if env_name in os.environ:
        return os.environ[env_name]

    # Check config
    spt = item.split(".")
    cfg: Any = config
    for s in spt:
        print(cfg)
        print(s)
        if cfg is None:
            return default
        try:
            cfg = getattr(cfg, s)
        except AttributeError:
            # TODO: Fancier error message for debugging
            raise ValueError(f"Config item {item} not found.")

    return cfg


def get_env_config_bool(config: Config | None, item: str | None, default: bool | None = None) -> bool:
    """
    Call get_env_config and convert strings to bools where makes sense.

    Throws an exception if the value is not a string and not convertable.
    """

    val = get_env_config(config, item, default)
    if val is None and default is not None:
        return default
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true" or val.lower() == "1"
    return bool(val)
