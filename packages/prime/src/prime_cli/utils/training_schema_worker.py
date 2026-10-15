"""Executed in the selected schema interpreter, without CLI dependencies."""

import json
import sys
from copy import deepcopy
from importlib import metadata
from uuid import uuid4


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


def explicit_fields(model, prefix=""):
    from pydantic import BaseModel

    fields = [prefix + name for name in model.model_fields_set]
    for name in type(model).model_fields:
        child = getattr(model, name)
        if isinstance(child, BaseModel):
            fields.extend(explicit_fields(child, prefix + name + "."))
    return sorted(fields)


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
    saved = defer_environments(data.get("orchestrator") or {}) if request["mode"] == "rl" else {}
    # Schema launchers normally provide SLURM settings and a run identity.
    if (data.get("deployment") or {}).get("type") == "multi_node" and "slurm" not in data:
        data["slurm"] = {}
    if "run" in model.model_fields:
        if "run" not in data:
            data["run"] = {"name": uuid4().hex}
        elif isinstance(data["run"], dict) and data["run"].get("name") is None:
            data["run"]["name"] = uuid4().hex
    config = model.model_validate(data)
    resolved = config.model_dump(mode="json")
    if request["mode"] == "sft":
        components = {"trainer": deepcopy(resolved)}
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
        restore_environments(components["orchestrator"], saved)
    # Only launcher metadata crosses this boundary twice; runtime ML fields
    # travel losslessly in components and remain opaque to Platform.
    deployment_settings = {key: resolved[key] for key in ("deployment", "output_dir", "clean")}
    deployment_settings["ckpt"] = (
        {"output_dir": resolved["ckpt"].get("output_dir")} if resolved.get("ckpt") else None
    )
    for key in ("weight_broadcast", "rollout_transport"):
        block = resolved.get(key)
        deployment_settings[key] = (
            {name: block[name] for name in ("type", "port") if name in block} if block else None
        )
    inference = resolved.get("inference")
    deployment_settings["inference"] = (
        {
            "deployment": inference.get("deployment"),
            "vllm": {
                "tensor_parallel_size": (inference.get("vllm") or {}).get("tensor_parallel_size")
            },
        }
        if inference
        else None
    )
    return {
        "contract_version": 4,
        "mode": request["mode"],
        "schema_commit": request["schema_commit"],
        "components": components,
        "resolved": deployment_settings,
        "run_dir": str(config.run_dir),
        "explicit_fields": explicit_fields(config),
    }


def main():
    try:
        request = json.load(sys.stdin)
        dist = metadata.distribution("prime-rl-configs")
        install = json.loads(dist.read_text("direct_url.json") or "{}")
        actual = install.get("vcs_info", {}).get("commit_id")
        expected = request["schema_commit"]
        if actual != expected:
            raise ValueError(f"schema provenance mismatch: expected {expected}, installed {actual}")
        result = {"artifact": normalize(request)}
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            error = "Config validation failed:\n" + "\n".join(
                f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                for item in exc.errors(include_input=False)
            )
        else:
            error = str(exc)
        print(json.dumps({"error": error}))
        sys.exit(1)


if __name__ == "__main__":
    main()
