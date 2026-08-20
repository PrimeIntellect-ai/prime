"""The config a run is actually configured with.

Two failures this covers, both of which produced a useless Config tab on the
platform: a resolved model dump that buries three chosen values under hundreds
of defaults, and a structured projection that cannot show the file someone
actually wrote.
"""

import json

import pytest
from conftest import RecordingHandler

import prime_runs as pr
from prime_runs.exceptions import ConfigurationError
from prime_runs.models import (
    CONFIG_SOURCE_KEY,
    MAX_CONFIG_SOURCE_BYTES,
    ConfigSource,
    RunSpec,
)
from prime_runs.run import _normalize_config

EVAL_TOML = """\
# the environment we are measuring
environment = "primeintellect/terminal-bench-2"
model = "deepseek/deepseek-v4-flash"

[env]
num_examples = 1
"""


# ------------------------------------------------------------------ coercion


def test_a_path_is_read_verbatim(tmp_path):
    """Comments, ordering and section grouping are the point — a dict loses all three."""
    path = tmp_path / "eval.toml"
    path.write_text(EVAL_TOML)

    source = ConfigSource.coerce(str(path))

    assert source.text == EVAL_TOML
    assert "# the environment we are measuring" in source.text
    assert source.format == "toml"
    assert source.filename == "eval.toml"


def test_the_format_is_inferred_from_the_suffix(tmp_path):
    for name, expected in [
        ("train.toml", "toml"),
        ("eval.json", "json"),
        ("run.yaml", "yaml"),
        ("run.yml", "yaml"),
        ("config", "text"),
    ]:
        path = tmp_path / name
        path.write_text("x = 1")
        assert ConfigSource.coerce(path).format == expected


def test_inline_text_has_to_be_explicit(tmp_path):
    """A bare string is a path. Guessing would turn a typo'd filename into a run
    whose config tab proudly displays the filename."""
    with pytest.raises(ConfigurationError, match="does not exist"):
        ConfigSource.coerce("environment = 'gsm8k'")

    source = ConfigSource.coerce(ConfigSource(text="environment = 'gsm8k'"))
    assert source.text == "environment = 'gsm8k'"


def test_an_oversized_file_is_refused_at_init_not_truncated(tmp_path):
    """Silently storing half a config is worse than not storing one."""
    path = tmp_path / "eval.toml"
    path.write_text("x = 1\n" * MAX_CONFIG_SOURCE_BYTES)

    with pytest.raises(ConfigurationError, match="over the"):
        ConfigSource.coerce(path)


def test_a_binary_file_is_refused(tmp_path):
    path = tmp_path / "eval.toml"
    path.write_bytes(b"\xff\xfe\x00binary")

    with pytest.raises(ConfigurationError, match="not UTF-8"):
        ConfigSource.coerce(path)


def test_an_unusable_type_says_what_was_expected():
    with pytest.raises(TypeError, match="config_source must be"):
        ConfigSource.coerce(object())


def test_a_mapping_round_trips():
    source = ConfigSource(text="a = 1", format="toml", filename="eval.toml")

    assert ConfigSource.from_mapping(source.to_dict()) == source


def test_a_mapping_without_text_is_not_a_config_source():
    assert ConfigSource.from_mapping({"format": "toml"}) is None
    with pytest.raises(ValueError, match="must contain a 'text' string"):
        ConfigSource.coerce({"format": "toml"})


# ------------------------------------------------------- config normalization


class FakeModel:
    """Duck-types the pydantic v2 surface ``_normalize_config`` looks for."""

    def __init__(self, set_fields, all_fields):
        self._set = set_fields
        self._all = all_fields

    def model_dump(self, mode=None, exclude_unset=False):
        return dict(self._set if exclude_unset else self._all)


def test_a_model_contributes_only_the_fields_someone_set():
    """The training Config tab's actual bug: ``exclude_none`` keeps every default,
    so three chosen values arrive buried in a hundred lines nobody picked."""
    model = FakeModel(
        set_fields={"model": "Qwen/Qwen3-8B", "max_steps": 1000},
        all_fields={"model": "Qwen/Qwen3-8B", "max_steps": 1000, "seed": 0, "log_level": "info"},
    )

    assert _normalize_config(model) == {"model": "Qwen/Qwen3-8B", "max_steps": 1000}


def test_a_mapping_is_taken_exactly_as_given():
    """The caller already decided what to say; second-guessing it would be worse."""
    assert _normalize_config({"a": 1, "b": None}) == {"a": 1, "b": None}


def test_a_mapping_is_copied_not_aliased():
    original = {"a": 1}
    assert _normalize_config(original) is not original


def test_no_config_is_an_empty_config():
    assert _normalize_config(None) == {}


def test_an_unusable_config_says_what_was_expected():
    with pytest.raises(TypeError, match="config must be a mapping"):
        _normalize_config(object())


# -------------------------------------------------------------- through init


def test_an_offline_run_stores_the_source_next_to_the_config(tmp_path):
    path = tmp_path / "eval.toml"
    path.write_text(EVAL_TOML)

    run = pr.init(
        name="tb2",
        environments=["gsm8k"],
        mode="offline",
        dir=str(tmp_path / "runs"),
        config={"num_examples": 1},
        config_source=path,
    )
    run.finish()

    state = json.loads((tmp_path / "runs" / run.id / "run.json").read_text())
    stored = state["spec"]["config"]
    # Both, not either: the structured form stays queryable, the source stays readable.
    assert stored["num_examples"] == 1
    assert stored[CONFIG_SOURCE_KEY]["text"] == EVAL_TOML
    assert stored[CONFIG_SOURCE_KEY]["filename"] == "eval.toml"


def test_the_run_reports_its_own_source(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(EVAL_TOML)

    run = pr.init(environments=["gsm8k"], mode="offline", dir=str(tmp_path), config_source=path)

    assert run.config_source.filename == "train.toml"
    assert run.config_source.text == EVAL_TOML
    run.finish()


def test_a_run_without_a_source_reports_none(tmp_path):
    run = pr.init(environments=["gsm8k"], mode="offline", dir=str(tmp_path))

    assert run.config_source is None
    run.finish()


def test_an_online_run_sends_the_source_in_create_metadata(
    tmp_path, monkeypatch, make_platform_client, eval_routes
):
    path = tmp_path / "eval.toml"
    path.write_text(EVAL_TOML)
    handler = RecordingHandler(eval_routes)
    monkeypatch.setattr("prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler))

    run = pr.init(
        name="tb2",
        environments=["gsm8k"],
        api_key="test-key",
        config={"num_examples": 1},
        config_source=path,
        sinks=[],
    )
    run.finish()

    create = next(r for r in handler.requests if r.url.path == "/api/v1/evaluations/")
    metadata = json.loads(create.content)["metadata"]
    assert metadata[CONFIG_SOURCE_KEY]["text"] == EVAL_TOML
    assert metadata["num_examples"] == 1


def test_the_source_survives_the_failure_fallback(tmp_path):
    """The fallback rewrites metadata to record a terminal state. It merges into
    the whole config, so the source must still be there afterwards."""
    path = tmp_path / "eval.toml"
    path.write_text(EVAL_TOML)

    run = pr.init(environments=["gsm8k"], mode="offline", dir=str(tmp_path), config_source=path)
    run.fail("something broke")

    state = json.loads((tmp_path / run.id / "run.json").read_text())
    assert state["config"][CONFIG_SOURCE_KEY]["text"] == EVAL_TOML


def test_a_spec_defaults_to_no_source():
    assert RunSpec().config == {}
