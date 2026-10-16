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


def reject_launcher_only_settings(config):
    if config.clean:
        raise ValueError("clean = true is unsupported for hosted runs: start a new run instead.")
    inference_deployment = getattr(config.inference, "deployment", None)
    if getattr(inference_deployment, "type", None) == "disaggregated":
        raise ValueError(
            'inference.deployment type = "disaggregated" is unsupported for hosted runs.'
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
        reject_launcher_only_settings(config)
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
