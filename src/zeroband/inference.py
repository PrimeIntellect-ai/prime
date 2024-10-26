from collections import defaultdict
import os

import torch
from pydantic_config import parse_argv
from einops import rearrange
from torch.nn import functional as F

from transformers import AutoTokenizer

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

import torch.distributed as dist
from zeroband import utils
from zeroband.comms import ElasticDeviceMesh
from zeroband.loss import cross_entropy_max_z_loss


from zeroband.data import TEST_VOCAB_SIZE, get_dataloader
from zeroband.models.llama import get_model
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
from zeroband.checkpoint import ModelWrapper

from zeroband.train import Config


def train(config: Config):
    # batch_size is the total batch size for all GPUs
    assert config.optim.batch_size % world_info.local_world_size == 0
    batch_size = config.optim.batch_size // world_info.local_world_size

    assert batch_size % config.train.micro_bs == 0
    gradient_accumulation_steps = batch_size // config.train.micro_bs

    if config.ckpt is not None and config.ckpt.interval is not None and config.diloco is not None:
        assert (
            config.ckpt.interval % config.diloco.inner_steps == 0
        ), "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"

    if config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    logger.debug("tokenizer loaded")

    train_dataloaders = get_dataloader(
        tokenizer=tokenizer,
        world_size=world_info.world_size,
        rank=world_info.rank,
        batch_size=config.train.micro_bs,
        data_config=config.data,
    )

    model, model_config = get_model(
        config.name_model,
        config.type_model,
        vocab_size=len(tokenizer) if config.name_model != "debugmodel" or not config.data.fake else TEST_VOCAB_SIZE,
        seq_length=config.data.seq_length,
        attn_fn=config.train.attn_fn,
    )

    model = model.to(world_info.local_rank)
    logger.debug("model loaded")

    num_params = utils.get_num_params(model, exclude_embedding=True)
    logger.info(f"Number of parameters: {num_params}")

    # if config.train.ac_ckpt:
    #     num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
    #     apply_ac_ckpt(model, num)

    elastic_device_mesh = ElasticDeviceMesh(enable=False)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32 if config.train.reduce_fp32 else None
    )

    for layer_id, transformer_block in model.layers.items():
        if config.train.reshard_after_forward:
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
        else:
            reshard_after_forward = False
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            mesh=elastic_device_mesh.cuda_local_mesh,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model,
        mp_policy=mp_policy,
        mesh=elastic_device_mesh.cuda_local_mesh,
        reshard_after_forward=config.train.reshard_after_forward,
    )

    if config.train.torch_compile:
        # we need to compile AFTER creating the CKPT manager, DON'T ASK ME WHY
        model = torch.compile(model)
        logger.debug("model compiled")

    if config.ckpt.resume is not None:
        # all is inplace
        states = {"model": ModelWrapper(model)}

        torch.distributed.checkpoint.load(states, checkpoint_id=config.ckpt.resume + "/diloco_0")

    loss_datasets = defaultdict(list)

    for name, train_dataloader in train_dataloaders.items():
        train_dataloader_iterator = iter(train_dataloader)

        for inner_step in range(config.optim.total_steps):
            loss_batch = 0
            z_loss_batch = 0

            for grad_acc_step in range(gradient_accumulation_steps):
                is_accumulating = grad_acc_step < gradient_accumulation_steps - 1
                # no sync if we are accumulating gradients
                model.set_requires_gradient_sync(not is_accumulating)

                batch = next(train_dataloader_iterator)
                input_ids = batch["input_ids"].to("cuda")
                labels = batch["labels"].to("cuda")
                if config.train.sequence_packing:
                    seqlens = batch["seqlens"].to("cuda")
                    # seqlens has a dynamic shape but fixed dimension, this allow to still torch compile
                    # https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html
                    torch._dynamo.mark_dynamic(seqlens, 0)
                else:
                    seqlens = None

                with torch.inference_mode():
                    logits = model(tokens=input_ids, seqlens=seqlens).contiguous()

                flatten_logits = rearrange(logits, "b seq vocab -> (b seq) vocab")
                flatten_labels = rearrange(labels, "b seq -> (b seq)")

                if config.optim.z_loss:
                    ce_loss, z_loss = cross_entropy_max_z_loss(
                        flatten_logits, flatten_labels, config.optim.z_loss_weight
                    )
                    ce_loss /= gradient_accumulation_steps
                    z_loss /= gradient_accumulation_steps

                    del logits
                    loss = ce_loss + z_loss

                else:
                    loss = F.cross_entropy(flatten_logits, flatten_labels) / gradient_accumulation_steps
                    del logits

                if config.optim.z_loss:
                    loss_batch += ce_loss.clone().detach()
                    z_loss_batch += z_loss.clone().detach()
                else:
                    loss_batch += loss.clone().detach()

            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg)

            logger.debug(f"loss: {loss_batch.item()}")

            loss_datasets[name].append(loss_batch.item())

    for name, loss_dataset in loss_datasets.items():
        logger.info(f"loss over {name}: {sum(loss_dataset)/len(loss_dataset)}")


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    world_info = get_world_info()
    logger = get_logger()

    torch.cuda.set_device(world_info.local_rank)

    config = Config(**parse_argv())
    logger.debug(f"config: {config.model_dump()}")

    train(config)
