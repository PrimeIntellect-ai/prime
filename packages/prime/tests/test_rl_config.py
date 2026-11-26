from pathlib import Path

import pytest
from prime_cli.commands.rl import (
    _load_config_file,
    _load_config_stack,
    _looks_like_lora,
    _split_overrides,
    _warn_legacy_full_finetune_type,
)


def test_load_toml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Config file not found"):
        _load_config_file(str(tmp_path / "missing.toml"))


def test_load_toml_rejects_invalid_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text("[model\nname = ")
    with pytest.raises(ValueError, match="Invalid TOML"):
        _load_config_file(str(config_path))


def test_load_toml_returns_raw_table(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('[model]\nname = "Qwen/Qwen3-0.6B"\n')
    assert _load_config_file(str(config_path)) == {"model": {"name": "Qwen/Qwen3-0.6B"}}


def test_load_toml_keeps_dotted_key_with_multiline_string(tmp_path: Path) -> None:
    # toml 0.10.2 dropped the `taskset.` prefix here and hoisted the key onto the env entry.
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        '[[train.env]]\nid = "x"\ntaskset.append_to_system_prompt = """\nline1\nline2"""\n'
    )
    assert _load_config_file(str(config_path)) == {
        "train": {"env": [{"id": "x", "taskset": {"append_to_system_prompt": "line1\nline2"}}]}
    }


def _write(path: Path, body: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return str(path)


@pytest.mark.parametrize(
    ("name", "body"),
    [
        (
            "rl.json",
            '{"model": {"name": "Qwen/Qwen3-0.6B"}, "trainer": {"optim": {"max_norm": null}}}',
        ),
        ("rl.yaml", "model:\n  name: Qwen/Qwen3-0.6B\ntrainer:\n  optim:\n    max_norm: null\n"),
        ("rl.yml", "model: {name: Qwen/Qwen3-0.6B}\ntrainer: {optim: {max_norm: null}}\n"),
    ],
)
def test_load_config_file_reads_json_and_yaml(tmp_path: Path, name: str, body: str) -> None:
    config_path = tmp_path / name
    config_path.write_text(body)
    # Unlike TOML, both keep an explicit null, as prime-rl's resolved configs rely on.
    assert _load_config_file(str(config_path)) == {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "trainer": {"optim": {"max_norm": None}},
    }


@pytest.mark.parametrize(
    ("name", "body", "message"),
    [
        ("rl.json", "{", "Invalid JSON"),
        ("rl.yaml", "model: [", "Invalid YAML"),
        ("rl.json", "[1, 2]", "must hold a table of settings, got list"),
        ("rl.yaml", "", "must hold a table of settings, got NoneType"),
    ],
)
def test_load_config_file_rejects_bad_json_and_yaml(
    tmp_path: Path, name: str, body: str, message: str
) -> None:
    config_path = tmp_path / name
    config_path.write_text(body)
    with pytest.raises(ValueError, match=message):
        _load_config_file(str(config_path))


def test_config_stack_mixes_formats(tmp_path: Path) -> None:
    toml = _write(tmp_path / "rl.toml", "[trainer.optim]\nlr = 1e-6\nmax_norm = 1.0\n")
    resolved = _write(tmp_path / "rl.json", '{"trainer": {"optim": {"max_norm": null}}}')
    assert _load_config_stack(["@", toml, "@", resolved])[0] == {
        "trainer": {"optim": {"lr": 1e-6, "max_norm": None}}
    }


def test_config_stack_merges_like_prime_rl(tmp_path: Path) -> None:
    model = _write(
        tmp_path / "model" / "rl.toml",
        'seq_len = 8\n[slurm]\n[trainer.optim]\ntype = "muon"\nlr = 1e-6\n'
        "[inference.vllm]\nmax_model_len = 8\n"
        '[[orchestrator.train.source]]\nname = "a"\n[[orchestrator.train.source]]\nname = "b"\n',
    )
    gpu = _write(
        tmp_path / "model" / "gpu" / "h200.toml",
        "[trainer.optim]\nlr = 2e-6\n[inference.vllm]\ntensor_parallel_size = 2\n",
    )
    data = _write(
        tmp_path / "data" / "swe.toml",
        '[[orchestrator.train.source]]\nname = "swe"\nenv.taskset.id = "scaleswe"\n',
    )
    fake = _write(tmp_path / "fake.toml", 'inference = "None"\n')

    merged, label = _load_config_stack([model, gpu, data])
    assert merged == {
        "seq_len": 8,
        "slurm": {},
        "trainer": {"optim": {"type": "muon", "lr": 2e-6}},
        "inference": {"vllm": {"max_model_len": 8, "tensor_parallel_size": 2}},
        # Arrays of tables replace rather than append, as in prime-rl.
        "orchestrator": {
            "train": {"source": [{"name": "swe", "env": {"taskset": {"id": "scaleswe"}}}]}
        },
    }
    assert label == f"{model} + {gpu} + {data}"
    # The "None" sentinel passes through for the schema to turn into null.
    assert _load_config_stack([model, fake])[0]["inference"] == "None"


def test_config_stack_accepts_prime_rl_at_syntax(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.toml", "x = 1\ny = 1\n")
    b = _write(tmp_path / "b.toml", "y = 2\n")
    expected = ({"x": 1, "y": 2}, f"{a} + {b}")
    assert _load_config_stack(["@", a, "@", b]) == expected
    assert _load_config_stack([f"@{a}", f"@{b}"]) == expected
    with pytest.raises(ValueError, match="at least one config file"):
        _load_config_stack(["@"])


def test_config_stack_resolves_env_files_per_file(tmp_path: Path) -> None:
    model = _write(tmp_path / "model" / "rl.toml", 'env_file = "model.env"\n')
    data = _write(tmp_path / "data" / "swe.toml", 'env_files = ["a.env", "b.env"]\n')
    merged, _ = _load_config_stack([model, data])
    assert merged == {
        "env_files": [
            str(tmp_path / "model" / "model.env"),
            str(tmp_path / "data" / "a.env"),
            str(tmp_path / "data" / "b.env"),
        ]
    }
    bad = _write(tmp_path / "bad.toml", "env_files = 1\n")
    with pytest.raises(ValueError, match="env_files in .*bad.toml must be a path"):
        _load_config_stack([bad])


def test_config_stack_rejects_conflicting_modes(tmp_path: Path) -> None:
    rl = _write(tmp_path / "rl.toml", 'mode = "rl"\n')
    same = _write(tmp_path / "same.toml", 'mode = "rl"\n')
    sft = _write(tmp_path / "sft.toml", 'mode = "sft"\n')
    assert _load_config_stack([rl, same])[0] == {"mode": "rl"}
    with pytest.raises(ValueError, match="conflicting modes"):
        _load_config_stack([rl, sft])


@pytest.mark.parametrize(
    ("args", "config_args", "overrides"),
    [
        (["a.toml", "b.toml"], ["a.toml", "b.toml"], []),
        (
            ["@", "a.toml", "@", "b.toml", "--trainer.model.debug.force-balanced-routing"],
            ["@", "a.toml", "@", "b.toml"],
            ["--trainer.model.debug.force-balanced-routing"],
        ),
        # A flag may take several values, so bare tokens after it are values.
        (
            ["a.toml", "--trainer.optim.lr", "-1e-3", "--items", "0.7", "b.toml"],
            ["a.toml"],
            ["--trainer.optim.lr", "-1e-3", "--items", "0.7", "b.toml"],
        ),
        # `@ file` after a flag's value is still a root config, as in prime-rl.
        (
            ["a.toml", "--deployment.num-infer-replicas=2", "--x", "1", "@", "b.toml", "@c.toml"],
            ["a.toml", "@", "b.toml", "@c.toml"],
            ["--deployment.num-infer-replicas=2", "--x", "1"],
        ),
    ],
)
def test_split_overrides(args, config_args, overrides) -> None:
    assert _split_overrides(args) == (config_args, overrides)


@pytest.mark.parametrize("args", [["a.toml", "--trainer", "@", "t.toml"], ["--trainer", "@t.toml"]])
def test_split_overrides_rejects_nested_config_files(args) -> None:
    with pytest.raises(ValueError, match="`--trainer @ file` is not supported"):
        _split_overrides(args)


def test_looks_like_lora_top_level_model_string() -> None:
    assert _looks_like_lora({"model": "Qwen/Qwen3-0.6B", "max_steps": 10}) is True


def test_looks_like_lora_top_level_env_array() -> None:
    assert _looks_like_lora({"env": [{"id": "primeintellect/reverse-text"}]}) is True


def test_looks_like_lora_prime_rl_config() -> None:
    cfg = {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "orchestrator": {"batch_size": 128},
        "trainer": {},
    }
    assert _looks_like_lora(cfg) is False


def test_warn_legacy_full_finetune_type_warns_on_leftover_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        "prime_cli.commands.rl.console.print",
        lambda *args, **_: printed.append(" ".join(str(a) for a in args)),
    )

    _warn_legacy_full_finetune_type({"type": "full_finetune"}, "rl.toml")

    assert any("type" in line and "no longer used" in line for line in printed)


def test_warn_legacy_full_finetune_type_silent_without_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        "prime_cli.commands.rl.console.print",
        lambda *args, **_: printed.append(" ".join(str(a) for a in args)),
    )

    _warn_legacy_full_finetune_type({"deployment": {"num_train_gpus": 1}}, "rl.toml")

    assert printed == []
