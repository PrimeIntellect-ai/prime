from enum import Enum
from typing import Literal, TypeAlias

from pydantic import model_validator
from pydantic_config import BaseConfig

class Compression(Enum):
    NO = "no"
    UINT8 = "uint8"


class DataConfig(BaseConfig):
    dataset_name_or_paths: str = "datasets/fineweb-edu"
    val_dataset_name_or_paths: str | None = None
    sequence_packing: bool = True
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
    type: Literal["adam"] = "adam"
    betas1: float = 0.9
    betas2: float = 0.95


class AdamWConfig(BaseConfig):
    type: Literal["adamw"] = "adamw"
    weight_decay: float = 0.1
    betas1: float = 0.9
    betas2: float = 0.95


class LearningRateSchedulerConfig(BaseConfig):
    decay_type: Literal["linear", "cosine", "sqrt"] = "linear"
    lr: float = 6e-4
    end_lr: float = 0.0
    num_decay_steps: int = 60000
    num_warmup_steps: int = 2000
    num_stable_steps: int = 0

    @property
    def num_total_steps(self):
        """
        The total number of steps that the learning rate scheduler defines in its current configuration.
        """
        return self.num_decay_steps + self.num_warmup_steps + self.num_stable_steps


# Union of all optimizer configuration types.
# New optimizer configurations must be added here to be picked up by the config system.
# Each configuration will be tried until a successful match is found.
# The 'type' field determines which class to use because the string literal is distinct for each class.
OptimizerConfig: TypeAlias = AdamConfig | AdamWConfig


class TrainConfig(BaseConfig):
    optimizer: OptimizerConfig = AdamConfig()
    batch_size: int = 512
    lr_scheduler: LearningRateSchedulerConfig = LearningRateSchedulerConfig()

class DilocoConfig(BaseConfig):
    outer_lr: float = 0.7
    inner_steps: int
    compression: Compression = Compression.NO

class MemoryProfilerConfig(BaseConfig):
    freq: int = 10
    snapshot_dir: str

AttnFnType: TypeAlias = Literal["flex", "math"]

class HardwareConfig(BaseConfig):
    micro_batch_size: int = 1

    act_ckpt: bool | int = False

    reshard_after_forward: bool = True  # old shard grad op True mean full shard

    reduce_fp32: bool = False  # should be True if SXM. Keep to false as default for backward compatibility

    log_model_hash: bool = False

    memory_profiler: MemoryProfilerConfig | None = None

    torch_profiler: bool = False

    torch_compile: bool = True

    fsdp_cpu_offload: bool = False

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
    resume: str | None = None


class Config(BaseConfig):
    # Project/Run
    project: str = "zeroband"
    run_id: str | None = None
    run_name: str | None = None

    # Model config
    model_name: Literal["debugmodel", "70M", "150M", "271M", "1B", "7B", "10B", "13B", "26B", "70B"] = "150M"
    model_type: Literal["llama2", "llama3"] = "llama3"

    # Logger
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"
    wandb_resume: bool = False
    log_level: Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_all_rank: bool = False

    # sub config
    diloco: DilocoConfig | None = None
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    hardware: HardwareConfig
    monitor: MonitorConfig | None = None

    ckpt: CkptConfig = CkptConfig()

    wandb: bool = True

    @model_validator(mode="after")
    def ckpt_diloco_step(self):
        if self.ckpt is not None and self.ckpt.interval is not None and self.diloco is not None:
            assert self.ckpt.interval % self.diloco.inner_steps == 0, (
                "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"
            )
        return self
