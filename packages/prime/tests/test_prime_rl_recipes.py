import json
from typing import Any, Dict

import httpx
import pytest
from prime_cli.commands.rl import _load_config_stack
from prime_cli.main import app
from prime_cli.utils import prime_rl_recipes
from prime_cli.utils.prime_rl_recipes import PrimeRLSource
from typer.testing import CliRunner

pytestmark = pytest.mark.usefixtures("stub_local_schema_validation")

SHA = "a" * 40

FILES: Dict[str, str] = {
    "recipes/m/rl.toml": 'seq_len = 1024\noutput_dir = "/x"\n[slurm]\n[model]\nname = "org/m"\n',
    "recipes/m/gpu/h200.toml": "[deployment]\nnum_train_nodes = 1\n",
    "recipes/multi/rl.toml": "seq_len = 1\n",
    "recipes/multi/gpu/b300.toml": "",
    "recipes/multi/gpu/gb200.toml": "",
    "recipes/m/gpu/experimental/b300.toml": "",
    "recipes/data/swe.toml": '[orchestrator]\nbatch_size = 8\n[env_vars]\nHF_HOME = "/h"\n',
    "recipes/data/math.toml": "",
    "recipes/bad/rl.toml": 'env_files = ["secrets.env"]\n',
    "README.md": "",
}


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def fake_get(url: str, **_: object) -> httpx.Response:
        calls.append(url)
        if url.endswith("/commits/main"):
            return httpx.Response(200, text=SHA)
        if "/commits/" in url:
            return httpx.Response(404)
        if "/git/trees/" in url:
            tree = [{"path": p, "type": "blob"} for p in FILES]
            return httpx.Response(200, json={"tree": tree})
        path = url.split(f"/{SHA}/", 1)[1]
        return httpx.Response(200, text=FILES[path])

    monkeypatch.setattr(prime_rl_recipes.httpx, "get", fake_get)
    return calls


def test_lists_recipes(github: list) -> None:
    recipes = {r.model: r for r in PrimeRLSource().recipes()}
    assert set(recipes) == {"m", "multi", "bad"}
    # Only direct children of gpu/ are overlays.
    assert recipes["m"].gpus == ["h200"]
    assert recipes["multi"].gpu is None
    assert not recipes["multi"].launchable
    assert recipes["m"].data == ["math", "swe"]


def test_recipe_expands_to_stack_and_drops_self_hosted_keys(github: list) -> None:
    merged, label = _load_config_stack(["recipe:m/swe"])
    assert merged == {
        "seq_len": 1024,
        "model": {"name": "org/m"},
        "deployment": {"num_train_nodes": 1},
        "orchestrator": {"batch_size": 8},
    }
    assert label == (
        "prime-rl@aaaaaaa:recipes/m/rl.toml + prime-rl@aaaaaaa:recipes/m/gpu/h200.toml"
        " + prime-rl@aaaaaaa:recipes/data/swe.toml"
    )
    # One commit lookup and one tree listing for the whole stack.
    assert sum("/commits/" in c for c in github) == 1
    assert sum("/git/trees/" in c for c in github) == 1


def test_local_file_overrides_recipe(github: list, tmp_path) -> None:
    local = tmp_path / "override.toml"
    local.write_text("seq_len = 2048\n")
    merged, _ = _load_config_stack(["recipe:m", str(local)])
    assert merged["seq_len"] == 2048
    assert "orchestrator" not in merged


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        ("recipe:nope", "Unknown recipe 'nope'"),
        ("recipe:m/nope", "Unknown data overlay 'nope'"),
        ("recipe:multi", "several GPU overlays"),
        ("recipe:bad", "env files can't come from the prime-rl repo"),
    ],
)
def test_errors(github: list, arg: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _load_config_stack([arg])


@pytest.mark.parametrize(
    "arg", ["recipe:", "recipe:m/", "recipe:/swe", "recipe:m/swe/x", "recipe:m#x"]
)
def test_malformed_spec_fails_before_any_request(github: list, arg: str) -> None:
    with pytest.raises(ValueError, match="expected recipe:<model>"):
        _load_config_stack([arg])
    assert github == []


runner = CliRunner()
TEST_ENV = {"PRIME_DISABLE_VERSION_CHECK": "1"}


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch) -> list:
    captured: list[dict[str, Any]] = []

    def fake_create_run(self: object, payload: dict) -> Any:
        from prime_cli.api.training import HostedTrainingRunResponse

        captured.append(payload)
        return HostedTrainingRunResponse(run_id="r1", token_value="t")

    monkeypatch.setattr("prime_cli.api.training.HostedTrainingClient.create_run", fake_create_run)
    return captured


def test_ref_flag_only_picks_runtime_code(github: list, dispatched: list) -> None:
    result = runner.invoke(
        app, ["train", "recipe:m/swe", "--ref", "my-code", "-y", "-o", "json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert github[0].endswith("/commits/main")
    assert dispatched[0]["sourceRef"] == "my-code"


def test_recipes_command_lists_launchable_recipes(github: list) -> None:
    result = runner.invoke(app, ["train", "recipes", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["commit"] == SHA
    assert [r["model"] for r in data["recipes"]] == ["bad", "m"]
    assert data["recipes"][1]["gpu"] == "h200"
