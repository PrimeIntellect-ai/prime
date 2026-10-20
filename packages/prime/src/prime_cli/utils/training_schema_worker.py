import json
import re
import sys
from copy import deepcopy
from importlib import metadata


def defer_environments(orch):
    saved = {}
    train = orch.get("train")
    if isinstance(train, dict) and "source" in train:
        saved["train_source"] = train.pop("source")
    if "eval" in orch:
        saved["eval"] = orch.pop("eval")
    return saved


def restore_environments(orch, saved):
    if "train_source" in saved:
        orch.setdefault("train", {})["source"] = saved["train_source"]
    if "eval" in saved:
        orch["eval"] = saved["eval"]


def check_deferred_settings(model, data, deferred, mode):
    """Validate deferred blocks without their environment sources, which need plugins."""
    probe = deepcopy(data)
    blocks = {"orchestrator.eval": deferred["eval"]} if "eval" in deferred else {}
    if mode == "sft":
        blocks = dict(deferred)
    for path, block in blocks.items():
        *parents, name = path.split(".")
        target = probe
        for parent in parents:
            target = target.setdefault(parent, {})
        target[name] = (
            {key: value for key, value in block.items() if key != "source"}
            if isinstance(block, dict)
            else block
        )
    if not blocks:
        return
    from pydantic import ValidationError

    try:
        model.model_validate(probe)
        return
    except ValidationError as exc:
        errors = exc.errors(include_input=False)
    except Exception:
        return
    prefixes = [tuple(path.split(".")) for path in blocks]
    problems = [
        error
        for error in errors
        if any(
            deferred_block_error(tuple(error["loc"]), error["type"], prefix) for prefix in prefixes
        )
    ]
    if problems:
        raise ValueError(format_errors(problems))


def deferred_block_error(loc, kind, block):
    if loc[: len(block)] != block:
        return False
    if len(loc) == len(block):
        # Validators on the block itself can need the removed sources; its type cannot.
        return kind not in ("value_error", "assertion_error")
    return loc[len(block)] != "source"


def format_errors(errors):
    return "Config validation failed:\n" + "\n".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors
    )


def reject_unsupported_hosted_settings(config):
    if config.clean:
        raise ValueError("clean = true is unsupported for hosted runs: start a new run instead.")
    inference_deployment = getattr(config.inference, "deployment", None)
    if getattr(inference_deployment, "type", None) == "disaggregated":
        raise ValueError(
            'inference.deployment type = "disaggregated" is unsupported for hosted runs.'
        )
    # Hosted runs start the orchestrator only alongside their own inference pods, so a
    # single-node run without inference GPUs is trainer-only, as multi-node runs with
    # num_infer_nodes = 0 are in prime-rl.
    deployment = config.deployment
    trainer_only = deployment.type == "single_node" and (
        config.inference is None or deployment.num_infer_gpus == 0
    )
    if trainer_only and getattr(getattr(config.trainer, "data", None), "fake", None) is None:
        raise ValueError(
            "Hosted RL runs need inference GPUs: set deployment.num_infer_gpus and an "
            "[inference] table, or use fake trainer data (trainer.data.fake) for a "
            "trainer-only run."
        )


def installed_commit(install):
    if "vcs_info" in install:
        return install["vcs_info"].get("commit_id")
    archive = re.fullmatch(
        r"https://github\.com/PrimeIntellect-ai/prime-rl/archive/([0-9a-f]{40})\.tar\.gz",
        install.get("url", ""),
    )
    if archive and install.get("subdirectory") == "packages/prime-rl-configs":
        return archive.group(1)
    return None


def normalize(request):
    data = deepcopy(request["config"])
    if data.pop("k8s", None):
        raise ValueError(
            "[k8s] is not supported for hosted runs: Platform manages deployment settings. "
            "Remove the [k8s] table."
        )
    from prime_rl.configs.rl import RLConfig
    from prime_rl.configs.sft import SFTConfig
    from prime_rl.utils import config as serialization

    model = RLConfig if request["mode"] == "rl" else SFTConfig
    if request["mode"] == "rl":
        deferred = defer_environments(data.get("orchestrator") or {})
    else:
        deferred = {key: data.pop(key) for key in request["sft_online_eval_keys"] if key in data}
    # Schema launchers normally provide SLURM settings.
    if (data.get("deployment") or {}).get("type") == "multi_node" and "slurm" not in data:
        data["slurm"] = {}
    config = model.model_validate(data)
    if request["mode"] == "rl":
        reject_unsupported_hosted_settings(config)
    check_deferred_settings(model, data, deferred, request["mode"])
    if request["mode"] == "sft":
        components = {"trainer": config.model_dump(mode="json") | deferred}
    else:
        dump = getattr(serialization, "dump_rl_components", None)
        if dump is not None:
            components = dump(config)
        else:
            components = {
                "trainer": config.trainer.model_dump(mode="json"),
                "orchestrator": config.orchestrator.model_dump(mode="json"),
            }
            if config.inference is not None:
                components["inference"] = config.inference.model_dump(
                    mode="json", exclude={"deployment", "slurm", "output_dir", "dry_run"}
                )
                if config.deployment.type == "multi_node":
                    components["inference"]["router"] = None
        restore_environments(components["orchestrator"], deferred)
    return {
        "contract_version": 4,
        "mode": request["mode"],
        "schema_commit": request["schema_commit"],
        "deployment": config.deployment.model_dump(mode="json"),
        "components": components,
    }


def main():
    try:
        request = json.load(sys.stdin)
        dist = metadata.distribution("prime-rl-configs")
        install = json.loads(dist.read_text("direct_url.json") or "{}")
        actual = installed_commit(install)
        expected = request["schema_commit"]
        if actual != expected:
            raise ValueError(f"schema provenance mismatch: expected {expected}, installed {actual}")
        result = {"artifact": normalize(request)}
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            error = format_errors(exc.errors(include_input=False))
        else:
            error = str(exc)
        print(json.dumps({"error": error}))
        sys.exit(1)


if __name__ == "__main__":
    main()
