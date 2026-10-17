from typing import Any

import pytest
from prime_cli.commands.sandbox import _lan_rule_text, _parse_lan_rule
from prime_cli.main import app
from prime_sandboxes.models import LanNetwork
from typer.testing import CliRunner

runner = CliRunner()


def test_parse_lan_rule_round_trips():
    rule = _parse_lan_rule("tag:web->tag:db@5432,6379")
    assert (rule.from_, rule.to, rule.ports) == ("tag:web", "tag:db", [5432, 6379])
    assert _lan_rule_text(rule) == "tag:web->tag:db@5432,6379"
    assert _parse_lan_rule("*->*").ports is None


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
