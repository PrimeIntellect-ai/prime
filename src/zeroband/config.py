from typing import Literal

from pydantic import model_validator
from pydantic_config import BaseConfig

from zeroband.checkpoint import CkptConfig
from zeroband.data import DataConfig
from zeroband.diloco import DilocoConfig
from zeroband.models.llama.model import AttnFnType
from zeroband.optimizers import OptimizersConfig, AdamConfig


class OptimConfig(BaseConfig):
    optim: OptimizersConfig = AdamConfig()

    lr: float = 4e-4
    weight_decay: float = 0.1
    adam_betas1: float = 0.9
    adam_betas2: float = 0.95

    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512

    z_loss: bool = False
    z_loss_weight: float = 2e-4
    num_chunks: int | None = None


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


class Config(BaseConfig):
    # main config
    name_model: Literal["debugmodel", "150M", "271M", "1B", "7B", "10B", "13B", "26B", "70B"] = "150M"
    type_model: Literal["llama2", "llama3"] = "llama3"

    project: str = "zeroband"
    run_id: str | None = None
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"
    wandb_resume: bool = False

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

