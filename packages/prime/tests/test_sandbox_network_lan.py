import pytest
from prime_cli.commands.sandbox import _lan_rule_text, _parse_lan_rule


def test_parse_lan_rule_round_trips():
    rule = _parse_lan_rule("tag:web->tag:db@5432,6379")
    assert (rule.from_, rule.to, rule.ports) == ("tag:web", "tag:db", [5432, 6379])
    assert _lan_rule_text(rule) == "tag:web->tag:db@5432,6379"
    assert _parse_lan_rule("*->*").ports is None


@pytest.mark.parametrize("raw", ["tag:web", "->tag:db", "*->*@http", "tag:web->tag:db@"])
def test_parse_lan_rule_rejects_malformed(raw):
    with pytest.raises(ValueError):
        _parse_lan_rule(raw)
