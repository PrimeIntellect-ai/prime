from enum import Enum
from typing import Literal, TypeAlias

from pydantic import model_validator
from pydantic_config import BaseConfig

AttnFnType: TypeAlias = Literal["flex", "math"]


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
    micro_bs: int = 1

    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard

    reduce_fp32: bool = False  # should be True if SXM. Keep to false as default for backward compatibility

    log_model_hash: bool = False

    memory_profiler: MemoryProfilerConfig | None = None

    torch_profiler: bool = False

    torch_compile: bool = True

    fused_linear_ce: bool = False

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
    # main config
    name_model: Literal["debugmodel", "70M", "150M", "271M", "1B", "7B", "10B", "13B", "26B", "70B"] = "150M"
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

    wandb: bool = True

    @model_validator(mode="after")
    def ckpt_diloco_step(self):
        if self.ckpt is not None and self.ckpt.interval is not None and self.diloco is not None:
            assert self.ckpt.interval % self.diloco.inner_steps == 0, (
                "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"
            )
        return self
