import json
from typing import Any, Optional

import pytest
from prime_cli.api.projects import Project
from prime_cli.core import APIError
from prime_cli.main import app
from prime_cli.utils.projects import (
    PROJECT_CONTEXT_CLEARED_KEY,
    PROJECT_CONTEXT_ENV_OVERRIDE_KEY,
    get_active_project_id,
    resolve_project_id,
    write_project_context,
)
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_API_KEY": "test-key",
    "PRIME_TEAM_ID": None,
}


def _project(team_id: Optional[str] = None) -> Project:
    return Project.model_validate(
        {
            "id": "cmproject0000000000000001",
            "name": "Battleship Baseline",
            "slug": "battleship-baseline",
            "status": "ACTIVE",
            "userId": "cmuser000000000000000001",
            "teamId": team_id,
            "createdAt": "2026-05-20T12:00:00Z",
            "updatedAt": "2026-05-20T12:00:00Z",
        }
    )


def test_project_create_sets_active_context(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def create(
            self,
            name: str,
            slug: Optional[str] = None,
            description: Optional[str] = None,
            team_id: Optional[str] = None,
        ) -> Project:
            assert name == "Battleship Baseline"
            assert slug is None
            assert description is None
            assert team_id is None
            return _project()

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(app, ["project", "create", "Battleship Baseline"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    context_path = tmp_path / ".prime" / "lab" / "context.json"
    assert context_path.exists()
    context_text = context_path.read_text()
    assert "cmproject0000000000000001" in context_text
    assert context_text.endswith("\n")


def test_project_create_team_project_requires_matching_active_team_to_use(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(
        app,
        ["project", "create", "Team Project", "--team-id", "team-123"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "Cannot create and set an active project for team team-123" in result.output
    assert not (tmp_path / ".prime" / "lab" / "context.json").exists()


def test_project_create_team_project_no_use_does_not_set_active_context(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def create(
            self,
            name: str,
            slug: Optional[str] = None,
            description: Optional[str] = None,
            team_id: Optional[str] = None,
        ) -> Project:
            assert name == "Team Project"
            assert slug is None
            assert description is None
            assert team_id == "team-123"
            return _project(team_id="team-123")

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(
        app,
        ["project", "create", "Team Project", "--team-id", "team-123", "--no-use"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert "Project created" in result.output
    assert not (tmp_path / ".prime" / "lab" / "context.json").exists()


def test_project_create_team_project_sets_active_when_cli_team_matches(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def create(
            self,
            name: str,
            slug: Optional[str] = None,
            description: Optional[str] = None,
            team_id: Optional[str] = None,
        ) -> Project:
            assert name == "Battleship Baseline"
            assert slug is None
            assert description is None
            assert team_id == "team-123"
            return _project(team_id="team-123")

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(
        app,
        ["project", "create", "Battleship Baseline"],
        env={**TEST_ENV, "PRIME_TEAM_ID": "team-123"},
    )

    assert result.exit_code == 0, result.output
    context_path = tmp_path / ".prime" / "lab" / "context.json"
    assert context_path.exists()
    assert '"team_id": "team-123"' in context_path.read_text()


def test_project_current_reads_active_context(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    context_dir = tmp_path / ".prime" / "lab"
    context_dir.mkdir(parents=True)
    (context_dir / "context.json").write_text(
        '{"project_id":"cmproject0000000000000001","team_id":null,'
        '"base_url":"https://api.primeintellect.ai"}'
    )

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "cmproject0000000000000001"
            assert team_id is None
            return _project()

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(app, ["project", "current"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Battleship Baseline" in result.output
    assert "battleship-baseline" in result.output


def test_project_current_json_api_error_keeps_project_shape_stable(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    context_dir = tmp_path / ".prime" / "lab"
    context_dir.mkdir(parents=True)
    (context_dir / "context.json").write_text(
        '{"project_id":"cmproject0000000000000001","project_slug":"battleship-baseline",'
        '"team_id":null,"base_url":"https://api.primeintellect.ai"}'
    )

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "cmproject0000000000000001"
            assert team_id is None
            raise APIError("offline")

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(app, ["project", "current", "--output", "json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["project"] is None
    assert payload["context"]["project_id"] == "cmproject0000000000000001"


def test_project_show_uses_active_context(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    context_dir = tmp_path / ".prime" / "lab"
    context_dir.mkdir(parents=True)
    (context_dir / "context.json").write_text(
        '{"project_id":"cmproject0000000000000001","team_id":null,'
        '"base_url":"https://api.primeintellect.ai"}'
    )

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "cmproject0000000000000001"
            assert team_id is None
            return _project()

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(app, ["project", "show"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Battleship Baseline" in result.output
    assert "Not set" in result.output


def test_project_show_explicit_ref_ignores_invalid_env_active_marker(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert team_id == "team-123"
            if project_ref == "team-project":
                return _project(team_id="other-team")
            assert project_ref == "battleship-baseline"
            return _project(team_id="team-123")

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)
    monkeypatch.setattr("prime_cli.utils.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(
        app,
        ["project", "show", "battleship-baseline"],
        env={**TEST_ENV, "PRIME_PROJECT_ID": "team-project", "PRIME_TEAM_ID": "team-123"},
    )

    assert result.exit_code == 0, result.output
    assert "Battleship Baseline" in result.output


def test_project_use_team_project_requires_matching_active_team(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "battleship-baseline"
            assert team_id == "team-123"
            return _project(team_id="team-123")

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(
        app,
        ["project", "use", "battleship-baseline", "--team-id", "team-123"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "Cannot set an active project for team team-123" in result.output
    assert not (tmp_path / ".prime" / "lab" / "context.json").exists()


def test_project_use_team_project_sets_active_when_cli_team_matches(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "battleship-baseline"
            assert team_id == "team-123"
            return _project(team_id="team-123")

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(
        app,
        ["project", "use", "battleship-baseline"],
        env={**TEST_ENV, "PRIME_TEAM_ID": "team-123"},
    )

    assert result.exit_code == 0, result.output
    assert "Active project updated" in result.output
    context_path = tmp_path / ".prime" / "lab" / "context.json"
    assert context_path.exists()
    assert '"team_id": "team-123"' in context_path.read_text()


def test_project_update_description(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def update(
            self,
            project_ref: str,
            name: Optional[str] = None,
            slug: Optional[str] = None,
            description: Optional[str] = None,
            team_id: Optional[str] = None,
        ) -> Project:
            assert project_ref == "battleship-baseline"
            assert name is None
            assert slug is None
            assert description == "Baseline and follow-up Battleship runs"
            assert team_id is None
            return _project().model_copy(
                update={"description": "Baseline and follow-up Battleship runs"}
            )

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)

    result = runner.invoke(
        app,
        [
            "project",
            "update",
            "battleship-baseline",
            "--description",
            "Baseline and follow-up Battleship runs",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert "Project updated" in result.output
    assert "Baseline and follow-up Battleship runs" in result.output


def test_project_update_requires_field(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(app, ["project", "update", "battleship-baseline"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Provide --name, --slug, --description, or --clear-description" in result.output


def test_project_clear_removes_active_context(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    context_dir = tmp_path / ".prime" / "lab"
    context_dir.mkdir(parents=True)
    context_path = context_dir / "context.json"
    context_path.write_text(
        '{"project_id":"cmproject0000000000000001","team_id":null,'
        '"base_url":"https://api.primeintellect.ai"}'
    )

    result = runner.invoke(app, ["project", "clear"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Active project cleared" in result.output
    assert not context_path.exists()


def test_project_clear_disables_env_project_for_workspace(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(
        app,
        ["project", "clear"],
        env={**TEST_ENV, "PRIME_PROJECT_ID": "cmproject0000000000000001"},
    )

    context_path = tmp_path / ".prime" / "lab" / "context.json"
    context = json.loads(context_path.read_text())
    assert result.exit_code == 0, result.output
    assert "Active project cleared" in result.output
    assert "PRIME_PROJECT_ID is set" in result.output
    assert context["project_id"] is None
    assert context[PROJECT_CONTEXT_CLEARED_KEY] is True
    monkeypatch.setenv("PRIME_PROJECT_ID", "cmproject0000000000000001")
    assert get_active_project_id() is None


def test_project_context_written_with_env_set_overrides_env_project(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.setenv("PRIME_PROJECT_ID", "env-project")

    write_project_context(_project())

    context = json.loads((tmp_path / ".prime" / "lab" / "context.json").read_text())
    assert context[PROJECT_CONTEXT_ENV_OVERRIDE_KEY] is True
    assert get_active_project_id() == "cmproject0000000000000001"


def test_env_project_id_is_validated_against_active_scope(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_PROJECT_ID", "team-project")
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    api_client = object()

    class DummyProjectsClient:
        def __init__(self, client: object) -> None:
            assert client is api_client

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "team-project"
            assert team_id == "team-123"
            return _project(team_id="other-team")

    monkeypatch.setattr("prime_cli.utils.projects.ProjectsClient", DummyProjectsClient)

    with pytest.raises(APIError, match="Cannot use PRIME_PROJECT_ID for team other-team"):
        resolve_project_id(None, client=api_client, use_active_project=True)


def test_env_project_id_validation_creates_api_client_when_needed(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_PROJECT_ID", "team-project")
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    api_client = object()
    monkeypatch.setattr("prime_cli.utils.projects.APIClient", lambda: api_client)

    class DummyProjectsClient:
        def __init__(self, client: object) -> None:
            assert client is api_client

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "team-project"
            assert team_id == "team-123"
            return _project(team_id="other-team")

    monkeypatch.setattr("prime_cli.utils.projects.ProjectsClient", DummyProjectsClient)

    with pytest.raises(APIError, match="Cannot use PRIME_PROJECT_ID for team other-team"):
        resolve_project_id(None, use_active_project=True)


def test_project_assign_rejects_env_project_outside_active_scope(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "team-project"
            assert team_id == "team-123"
            return _project(team_id="other-team")

    class DummyRLClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def update_run_project(
            self,
            run_id: str,
            project_id: Optional[str],
            *,
            operation: str = "set",
            move_adapters: bool = True,
        ) -> tuple[object, int]:
            raise AssertionError("should not attach a run with an invalid env project")

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)
    monkeypatch.setattr("prime_cli.utils.projects.ProjectsClient", DummyProjectsClient)
    monkeypatch.setattr("prime_cli.commands.projects.RLClient", DummyRLClient)

    result = runner.invoke(
        app,
        ["project", "assign", "run", "run-123"],
        env={**TEST_ENV, "PRIME_PROJECT_ID": "team-project", "PRIME_TEAM_ID": "team-123"},
    )

    assert result.exit_code == 1
    assert "Cannot use PRIME_PROJECT_ID for team other-team" in result.output


def test_project_assign_run_uses_active_project(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    context_dir = tmp_path / ".prime" / "lab"
    context_dir.mkdir(parents=True)
    (context_dir / "context.json").write_text(
        '{"project_id":"cmproject0000000000000001","team_id":null,'
        '"base_url":"https://api.primeintellect.ai"}'
    )

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "cmproject0000000000000001"
            assert team_id is None
            return _project()

    class DummyRLClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def update_run_project(
            self,
            run_id: str,
            project_id: Optional[str],
            *,
            operation: str = "set",
            move_adapters: bool = True,
        ) -> tuple[object, int]:
            assert run_id == "run-123"
            assert project_id == "cmproject0000000000000001"
            assert operation == "add"
            assert move_adapters is True
            return object(), 2

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)
    monkeypatch.setattr("prime_cli.commands.projects.RLClient", DummyRLClient)

    result = runner.invoke(app, ["project", "assign", "run", "run-123"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Project assigned" in result.output
    assert "Adapters Updated" in result.output


def test_project_remove_run_forwards_targeted_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    captured = {}

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "battleship-baseline"
            assert team_id is None
            return _project()

    class DummyRLClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def update_run_project(
            self,
            run_id: str,
            project_id: Optional[str],
            *,
            operation: str = "set",
            move_adapters: bool = True,
        ) -> tuple[object, int]:
            captured.update(
                {
                    "run_id": run_id,
                    "project_id": project_id,
                    "operation": operation,
                    "move_adapters": move_adapters,
                }
            )
            return object(), 0

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)
    monkeypatch.setattr("prime_cli.commands.projects.RLClient", DummyRLClient)

    result = runner.invoke(
        app,
        [
            "project",
            "remove",
            "run",
            "run-123",
            "battleship-baseline",
            "--no-move-adapters",
            "--output",
            "json",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "run_id": "run-123",
        "project_id": "cmproject0000000000000001",
        "operation": "remove",
        "move_adapters": False,
    }


def test_project_assign_adapter_forwards_project_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    captured = {}

    class DummyProjectsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def get(self, project_ref: str, team_id: Optional[str] = None) -> Project:
            assert project_ref == "battleship-baseline"
            assert team_id is None
            return _project()

    class DummyDeploymentsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def update_adapter_project(
            self,
            adapter_id: str,
            project_id: Optional[str],
            *,
            operation: str = "set",
        ) -> object:
            captured.update(
                {
                    "adapter_id": adapter_id,
                    "project_id": project_id,
                    "operation": operation,
                }
            )
            return object()

    monkeypatch.setattr("prime_cli.commands.projects.ProjectsClient", DummyProjectsClient)
    monkeypatch.setattr("prime_cli.commands.projects.DeploymentsClient", DummyDeploymentsClient)

    result = runner.invoke(
        app,
        [
            "project",
            "assign",
            "adapter",
            "adapter-123",
            "battleship-baseline",
            "--output",
            "json",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "adapter_id": "adapter-123",
        "project_id": "cmproject0000000000000001",
        "operation": "add",
    }


def test_project_remove_adapter_without_project_clears_memberships(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    captured = {}

    class DummyDeploymentsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def update_adapter_project(
            self,
            adapter_id: str,
            project_id: Optional[str],
            *,
            operation: str = "set",
        ) -> object:
            captured.update(
                {
                    "adapter_id": adapter_id,
                    "project_id": project_id,
                    "operation": operation,
                }
            )
            return object()

    monkeypatch.setattr("prime_cli.commands.projects.DeploymentsClient", DummyDeploymentsClient)

    result = runner.invoke(
        app,
        ["project", "remove", "adapter", "adapter-123", "--output", "json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "adapter_id": "adapter-123",
        "project_id": None,
        "operation": "clear",
    }


def test_project_remove_eval_clears_project(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    class DummyEvalsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def update_evaluation(
            self,
            evaluation_id: str,
            *,
            clear_project: bool = False,
        ) -> dict:
            assert evaluation_id == "eval-123"
            assert clear_project is True
            return {"evaluation_id": evaluation_id}

    monkeypatch.setattr("prime_cli.commands.projects.EvalsClient", DummyEvalsClient)

    result = runner.invoke(app, ["project", "remove", "eval", "eval-123"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "Project removed" in result.output


def test_project_remove_eval_rejects_targeted_project(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    monkeypatch.setattr("prime_cli.commands.projects.APIClient", lambda: object())

    result = runner.invoke(
        app,
        ["project", "remove", "eval", "eval-123", "battleship-baseline"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "Targeted removal from one project is not supported for evaluations" in result.output


def test_active_project_context_is_discovered_from_parent_workspace(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("PRIME_API_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_PROJECT_ID", raising=False)
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    context_dir = tmp_path / ".prime" / "lab"
    context_dir.mkdir(parents=True)
    (tmp_path / ".prime" / "lab.json").write_text("{}")
    (context_dir / "context.json").write_text(
        '{"project_id":"cmproject0000000000000001","team_id":null,'
        '"base_url":"https://api.primeintellect.ai"}'
    )

    nested = tmp_path / "outputs" / "evals" / "gsm8k" / "run-123"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert get_active_project_id() == "cmproject0000000000000001"


def test_active_project_context_does_not_cross_lab_workspace_boundary(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("PRIME_API_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_PROJECT_ID", raising=False)
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    parent_context_dir = tmp_path / ".prime" / "lab"
    parent_context_dir.mkdir(parents=True)
    (tmp_path / ".prime" / "lab.json").write_text("{}")
    (parent_context_dir / "context.json").write_text(
        '{"project_id":"parent-project","team_id":null,"base_url":"https://api.primeintellect.ai"}'
    )

    child_workspace = tmp_path / "child-workspace"
    (child_workspace / ".prime").mkdir(parents=True)
    (child_workspace / ".prime" / "lab.json").write_text("{}")
    nested = child_workspace / "outputs" / "evals" / "gsm8k" / "run-123"
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert get_active_project_id() is None


def test_write_project_context_uses_parent_lab_workspace(monkeypatch, tmp_path) -> None:
    (tmp_path / ".prime" / "lab").mkdir(parents=True)
    (tmp_path / ".prime" / "lab.json").write_text("{}")
    nested = tmp_path / "outputs" / "evals" / "gsm8k" / "run-123"
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)
    monkeypatch.setenv("HOME", str(tmp_path))

    write_project_context(_project())

    assert (tmp_path / ".prime" / "lab" / "context.json").exists()
    assert not (nested / ".prime" / "lab" / "context.json").exists()


def test_project_and_no_project_are_mutually_exclusive() -> None:
    with pytest.raises(APIError, match="Cannot use --project and --no-project together"):
        resolve_project_id("cmproject0000000000000001", no_project=True)
