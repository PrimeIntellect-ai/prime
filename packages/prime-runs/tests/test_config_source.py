"""The config a run is actually configured with: the file someone wrote, kept
byte for byte, rather than a projection that cannot show it."""

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


# ------------------------------------------------------ the config-file form


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
    with pytest.raises(TypeError, match="config source must be"):
        ConfigSource.coerce(object())


def test_a_mapping_round_trips():
    source = ConfigSource(text="a = 1", format="toml", filename="eval.toml")

    assert ConfigSource.from_mapping(source.to_dict()) == source


def test_a_mapping_without_text_is_not_a_config_source():
    assert ConfigSource.from_mapping({"format": "toml"}) is None
    with pytest.raises(ValueError, match="must contain a 'text' string"):
        ConfigSource.coerce({"format": "toml"})


# ------------------------------------------------------- config normalization


def test_a_mapping_is_taken_exactly_as_given():
    """The caller already decided what to say; second-guessing it would be worse."""
    assert _normalize_config({"a": 1, "b": None}) == {"a": 1, "b": None}


def test_a_mapping_is_copied_not_aliased():
    original = {"a": 1}
    assert _normalize_config(original) is not original


def test_no_config_is_an_empty_config():
    assert _normalize_config(None) == {}


def test_an_unusable_config_says_what_was_expected():
    with pytest.raises(TypeError, match="config must be a path"):
        _normalize_config(object())


def test_a_path_becomes_a_stored_source(tmp_path):
    """One parameter, three forms — the same shape ``environments=`` already has."""
    path = tmp_path / "eval.toml"
    path.write_text(EVAL_TOML)

    assert _normalize_config(path) == {
        CONFIG_SOURCE_KEY: {"format": "toml", "text": EVAL_TOML, "filename": "eval.toml"}
    }
    assert _normalize_config(str(path))[CONFIG_SOURCE_KEY]["text"] == EVAL_TOML
    assert _normalize_config(ConfigSource(text="a = 1"))[CONFIG_SOURCE_KEY]["text"] == "a = 1"


def test_a_mapping_that_looks_like_a_source_is_still_just_a_mapping():
    """Form is decided by type, never by inspecting keys — so a config that
    happens to have a ``text`` field is not mistaken for a launch file."""
    assert _normalize_config({"text": "hello", "format": "toml"}) == {
        "text": "hello",
        "format": "toml",
    }


# -------------------------------------------------------------- through init


@pytest.fixture
def online(monkeypatch, make_platform_client, eval_routes):
    handler = RecordingHandler(eval_routes)
    monkeypatch.setattr("prime_runs.run.PlatformClient", lambda **_: make_platform_client(handler))
    monkeypatch.setattr("prime_traces.TracesClient", lambda **_: object())

    def _init(**kwargs):
        return pr.init(name="tb2", environments=["gsm8k"], api_key="test-key", **kwargs), handler

    return _init


def test_an_online_run_sends_the_source_in_create_metadata(tmp_path, online):
    path = tmp_path / "eval.toml"
    path.write_text(EVAL_TOML)

    run, handler = online(config=path)
    run.finish()

    create = next(r for r in handler.requests if r.url.path == "/api/v1/evaluations/")
    metadata = json.loads(create.content)["metadata"]
    assert metadata[CONFIG_SOURCE_KEY]["text"] == EVAL_TOML
    assert metadata[CONFIG_SOURCE_KEY]["format"] == "toml"
    assert metadata[CONFIG_SOURCE_KEY]["filename"] == "eval.toml"


def test_extra_values_can_be_merged_onto_a_launch_file(tmp_path, online):
    """A run launched from a file that also wants structured values passes a
    mapping carrying the source under ``CONFIG_SOURCE_KEY`` — what verifiers does."""
    path = tmp_path / "eval.toml"
    path.write_text(EVAL_TOML)
    config = {
        "model": "deepseek/deepseek-v4-flash",
        CONFIG_SOURCE_KEY: ConfigSource.from_file(path).to_dict(),
    }

    run, handler = online(config=config)
    run.finish()

    create = next(r for r in handler.requests if r.url.path == "/api/v1/evaluations/")
    metadata = json.loads(create.content)["metadata"]
    assert metadata["model"] == "deepseek/deepseek-v4-flash"
    assert metadata[CONFIG_SOURCE_KEY]["text"] == EVAL_TOML
    assert run.config_source.filename == "eval.toml"


def test_the_run_reports_its_own_source(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(EVAL_TOML)

    run = pr.init(environments=["gsm8k"], mode="disabled", config=path)

    assert run.config_source.filename == "train.toml"
    assert run.config_source.text == EVAL_TOML
    run.finish()


def test_a_run_without_a_source_reports_none():
    run = pr.init(environments=["gsm8k"], mode="disabled")

    assert run.config_source is None
    run.finish()


def test_the_source_survives_the_failure_fallback(tmp_path, online):
    """The fallback rewrites metadata to record a terminal state. It merges into
    the whole config, so the source must still be there afterwards."""
    path = tmp_path / "eval.toml"
    path.write_text(EVAL_TOML)

    run, handler = online(config=path)
    run.fail("something broke")

    update = handler.bodies_for("/api/v1/evaluations/eval-abc")[-1]
    assert update["metadata"][CONFIG_SOURCE_KEY]["text"] == EVAL_TOML
    assert update["metadata"]["prime_runs"]["status"] == "failed"


def test_a_spec_defaults_to_no_source():
    assert RunSpec().config == {}
