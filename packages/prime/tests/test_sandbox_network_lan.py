from typing import Any

import pytest
from prime_cli.commands.sandbox import _lan_rule_text, _parse_lan_rule, _parse_localhost_forward
from prime_cli.main import app
from prime_sandboxes.models import LanNetwork, LocalhostForwardsStatus
from typer.testing import CliRunner

runner = CliRunner()


def test_parse_lan_rule_round_trips():
    rule = _parse_lan_rule("tag:web->tag:db@5432,6379")
    assert (rule.from_, rule.to, rule.ports) == ("tag:web", "tag:db", [5432, 6379])
    assert _lan_rule_text(rule) == "tag:web->tag:db@5432,6379"
    assert _parse_lan_rule("*->*").ports is None
    assert _lan_rule_text(_parse_lan_rule("internet->tag:web@8080")) == "internet->tag:web@8080"


@pytest.mark.parametrize("raw", ["tag:web", "->tag:db", "*->*@http", "tag:web->tag:db@"])
def test_parse_lan_rule_rejects_malformed(raw):
    with pytest.raises(ValueError):
        _parse_lan_rule(raw)


def test_network_create_passes_its_rules(monkeypatch):
    seen: list[Any] = []

    def create_lan_network(self: Any, name: str, team_id: Any = None, rules: Any = None):
        seen.append((name, rules))
        return LanNetwork.model_validate(
            {
                "id": "net1",
                "name": name,
                "rules": [r.model_dump(by_alias=True) for r in rules or []],
                "version": 1,
                "synced": True,
                "created_at": "2026-10-17T00:00:00Z",
            }
        )

    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.create_lan_network", create_lan_network
    )
    result = runner.invoke(
        app, ["sandbox", "network", "create", "lab", "-r", "tag:web->tag:db@5432"]
    )
    assert result.exit_code == 0, result.output
    ((name, rules),) = seen
    assert (name, [_lan_rule_text(r) for r in rules]) == ("lab", ["tag:web->tag:db@5432"])
    assert "tag:web->tag:db@5432" in result.output
    # Without --rule the platform's default applies.
    seen.clear()
    assert runner.invoke(app, ["sandbox", "network", "create", "lab"]).exit_code == 0
    assert seen == [("lab", None)]


@pytest.mark.parametrize("raw", ["8000", ":sbx", "http:sbx", "0:sbx", "70000:sbx", "8000:"])
def test_parse_localhost_forward_rejects_malformed(raw):
    with pytest.raises(ValueError):
        _parse_localhost_forward(raw)


def test_network_localhost_replaces_shows_and_clears(monkeypatch):
    seen: list[Any] = []

    def status(forwards: Any) -> LocalhostForwardsStatus:
        return LocalhostForwardsStatus(
            forwards=forwards, generation=2, applied_generation=1, applied=False
        )

    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.set_localhost_forwards",
        lambda self, sandbox_id, forwards: seen.append((sandbox_id, forwards)) or status(forwards),
    )
    monkeypatch.setattr(
        "prime_cli.commands.sandbox.SandboxClient.get_localhost_forwards",
        lambda self, sandbox_id: status([]),
    )
    result = runner.invoke(
        app, ["sandbox", "network", "localhost", "sbx-app", "8000:sbx-api", "5432:sbx-db"]
    )
    assert result.exit_code == 0, result.output
    ((sandbox_id, forwards),) = seen
    assert sandbox_id == "sbx-app"
    assert [(f.port, f.sandbox_id) for f in forwards] == [(8000, "sbx-api"), (5432, "sbx-db")]
    assert "sbx-db.sandbox.internal:5432" in result.output
    assert "not active yet" in result.output

    result = runner.invoke(app, ["sandbox", "network", "localhost", "sbx-app"])
    assert result.exit_code == 0 and "No localhost forwards" in result.output

    seen.clear()
    assert (
        runner.invoke(app, ["sandbox", "network", "localhost", "sbx-app", "--clear"]).exit_code == 0
    )
    assert seen == [("sbx-app", [])]

    result = runner.invoke(app, ["sandbox", "network", "localhost", "sbx-app", "80:a", "80:b"])
    assert result.exit_code == 1 and "only once" in result.output
