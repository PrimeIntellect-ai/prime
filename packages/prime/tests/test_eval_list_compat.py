from prime_cli.commands import evals as evals_cmd


class _OldEvalsClient:
    def __init__(self) -> None:
        self.called_kwargs = None

    def list_evaluations(self, env_name=None, suite_id=None, skip=0, limit=50):
        self.called_kwargs = {
            "env_name": env_name,
            "suite_id": suite_id,
            "skip": skip,
            "limit": limit,
        }
        return {"evaluations": [], "total": 0}


class _NewEvalsClient:
    def __init__(self) -> None:
        self.called_kwargs = None

    def list_evaluations(self, env_name=None, suite_id=None, skip=0, limit=50, *, team_id=None):
        self.called_kwargs = {
            "env_name": env_name,
            "suite_id": suite_id,
            "skip": skip,
            "limit": limit,
            "team_id": team_id,
        }
        return {"evaluations": [], "total": 0}


class _ConfigWithTeam:
    team_id = "team-123"


class _ConfigNoTeam:
    team_id = None


def test_list_evals_falls_back_when_client_does_not_support_team_id(monkeypatch):
    client = _OldEvalsClient()
    captured = {}

    monkeypatch.setattr(evals_cmd, "APIClient", lambda: object())
    monkeypatch.setattr(evals_cmd, "Config", lambda: _ConfigWithTeam())
    monkeypatch.setattr(evals_cmd, "EvalsClient", lambda _api: client)
    monkeypatch.setattr(evals_cmd, "output_data_as_json", lambda data, _console: captured.update(data))

    evals_cmd.list_evals(output="json", num=10, page=3, env="hubert-marek/agent-diff-bench")

    assert captured == {"evaluations": [], "total": 0}
    assert client.called_kwargs == {
        "env_name": "hubert-marek/agent-diff-bench",
        "suite_id": None,
        "skip": 20,
        "limit": 10,
    }


def test_list_evals_passes_team_id_when_client_supports_it(monkeypatch):
    client = _NewEvalsClient()
    captured = {}

    monkeypatch.setattr(evals_cmd, "APIClient", lambda: object())
    monkeypatch.setattr(evals_cmd, "Config", lambda: _ConfigNoTeam())
    monkeypatch.setattr(evals_cmd, "EvalsClient", lambda _api: client)
    monkeypatch.setattr(evals_cmd, "output_data_as_json", lambda data, _console: captured.update(data))

    evals_cmd.list_evals(output="json", num=50, page=1, env="hubert-marek/agent-diff-bench")

    assert captured == {"evaluations": [], "total": 0}
    assert client.called_kwargs == {
        "env_name": "hubert-marek/agent-diff-bench",
        "suite_id": None,
        "skip": 0,
        "limit": 50,
        "team_id": None,
    }
