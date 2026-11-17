import json
from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_API_KEY": "test-key",
}
RUN = {
    "id": "run1",
    "pipeline_id": "p1",
    "version": "a" * 64,
    "package": "acme/triage@0.2.0",
    "status": "RUNNING",
    "error": None,
    "stopped": False,
    "outputs": [{"name": "report.md", "node": "report", "bytes": 3}],
    "state": {
        "jobs": {"terminal": 3, "held": 1},
        "stages": {"scout": {"terminal": 3}, "review": {"held": 1}},
        "waiting_on_people": [
            {"job": "batch", "stage": "review", "reason": "Keep these ideas?", "by": "review"}
        ],
        "failures": [],
    },
    "created_at": "2026-11-05T00:00:00",
}


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    calls: Dict[str, Any] = {}

    def post(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        calls["post"] = (endpoint, json)
        return RUN

    def get(self: Any, endpoint: str, params: Any = None, timeout: Any = None) -> Dict[str, Any]:
        calls["get"] = endpoint
        return RUN

    monkeypatch.setattr("prime_cli.client.APIClient.post", post)
    monkeypatch.setattr("prime_cli.client.APIClient.get", get)
    return calls


def test_run_submits_the_package_with_the_secrets_it_names(api: Dict[str, Any]) -> None:
    env = {**TEST_ENV, "GITHUB_TOKEN": "gh"}
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage@0.2.0", "--secret", "GITHUB_TOKEN"], env=env
    )
    assert result.exit_code == 0, result.output
    endpoint, body = api["post"]
    assert endpoint == "/pipelines/runs"
    assert body["package"] == "acme/triage@0.2.0"
    assert body["config"] == {} and body["capabilities"] == {}
    assert body["secrets"] == {"GITHUB_TOKEN": "gh"} and body["network"] == {"allow": []}
    assert body["pipeline_id"] is None and body["run_package"] is False
    assert body["name"] is None
    assert "Package: acme/triage@0.2.0" in result.output
    assert "prime pipelines get run1" in result.output


def test_run_names_the_pipeline_it_saves(api: Dict[str, Any]) -> None:
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--name", "nightly", "-t", "t1"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    body = api["post"][1]
    assert (body["name"], body["team_id"], body["pipeline_id"]) == ("nightly", "t1", None)
    assert "Pipeline: p1" in result.output


LAUNCH_TOML = """
[config]
items = 3
ask = true

[capabilities]
spend_limit = 20

[capabilities.publish.release]
name = "triage-tasks"
version = "0.1.0"

[capabilities.traces]
personal = true
"""
LAUNCH = {
    "config": {"items": 3, "ask": True},
    "capabilities": {
        "spend_limit": 20,
        "publish": {"release": {"name": "triage-tasks", "version": "0.1.0"}},
        "traces": {"personal": True},
    },
}


@pytest.mark.parametrize("name", ["launch.toml", "launch.json", "launch"])
def test_run_sends_the_launch_files_config_and_capabilities(
    api: Dict[str, Any], tmp_path, name
) -> None:
    file = tmp_path / name
    file.write_text(json.dumps(LAUNCH) if name.endswith(".json") else LAUNCH_TOML)
    hosts = ["--allow-host", "api.github.com", "--allow-host", "*.example.com"]
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--launch", str(file), *hosts], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    body = api["post"][1]
    assert (body["config"], body["capabilities"]) == (LAUNCH["config"], LAUNCH["capabilities"])
    assert body["network"] == {"allow": ["api.github.com", "*.example.com"]}


def test_run_sets_the_spend_limit_over_the_launch_files(api: Dict[str, Any], tmp_path) -> None:
    file = tmp_path / "launch.toml"
    file.write_text(LAUNCH_TOML)
    result = runner.invoke(
        app,
        ["pipelines", "run", "acme/triage", "--launch", str(file), "--spend-limit", "5"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    assert api["post"][1]["capabilities"] == {**LAUNCH["capabilities"], "spend_limit": 5.0}
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--spend-limit", "5"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert api["post"][1]["capabilities"] == {"spend_limit": 5.0}


@pytest.mark.parametrize(
    ("name", "text", "error"),
    [
        ("launch.toml", "[config\n", "is not TOML"),
        ("launch.json", "{", "is not JSON"),
        ("launch.json", "[]", "two tables, config and capabilities"),
        ("launch.toml", 'package = "acme/triage"\n', "two tables, config and capabilities"),
        ("launch.toml", "config = 3\n", "config is a table"),
    ],
)
def test_run_refuses_a_launch_file_it_cannot_send(
    api: Dict[str, Any], tmp_path, name, text, error
) -> None:
    file = tmp_path / name
    file.write_text(text)
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--launch", str(file)], env=TEST_ENV
    )
    assert result.exit_code == 1 and error in result.output
    assert "Traceback" not in result.output and "post" not in api


TEAM_RUN = (
    "This team run would, as you: publish triage-tasks@0.1.0 to the Hub (release); read your "
    "personal traces. Agree to run the package and what it declares with run_package."
)


@pytest.fixture
def team_run(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """A team run's start: refused, with what it would do, until its starter agrees."""
    from prime_cli.client import APIError

    calls: Dict[str, Any] = {"posts": [], "detail": TEAM_RUN}

    def post(self: Any, endpoint: str, json: Any = None, timeout: Any = None) -> Dict[str, Any]:
        calls["posts"].append(json)
        if not json["run_package"] or calls.get("always"):
            raise APIError(f"HTTP 422: {calls['detail']}")
        return RUN

    monkeypatch.setattr("prime_cli.client.APIClient.post", post)
    return calls


def test_a_team_run_shows_what_it_would_do_and_asks_once(team_run: Dict[str, Any]) -> None:
    env = {**TEST_ENV, "COLUMNS": "2000"}
    result = runner.invoke(app, ["pipelines", "run", "acme/triage"], env=env, input="y\n")
    assert result.exit_code == 0, result.output
    assert "publish triage-tasks@0.1.0 to the Hub (release); read your personal traces" in (
        result.output
    )
    assert "Agree, and start the run?" in result.output
    assert "To agree without being asked, pass --run-package." in result.output
    assert [body["run_package"] for body in team_run["posts"]] == [False, True]
    assert "prime pipelines get run1" in result.output


def test_a_team_run_not_agreed_to_starts_nothing(team_run: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "run", "acme/triage"], env=TEST_ENV, input="n\n")
    assert result.exit_code == 0 and "Cancelled." in result.output
    assert [body["run_package"] for body in team_run["posts"]] == [False]
    # No answer to read: it ends, rather than waiting, and nothing starts.
    team_run["posts"].clear()
    result = runner.invoke(app, ["pipelines", "--plain", "run", "acme/triage"], env=TEST_ENV)
    assert result.exit_code == 1
    assert [body["run_package"] for body in team_run["posts"]] == [False]


def test_run_package_agrees_without_being_asked(team_run: Dict[str, Any]) -> None:
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--run-package", "-o", "json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "run1"
    assert [body["run_package"] for body in team_run["posts"]] == [True]
    # Agreed to already: a refusal is about something else, and nothing is asked.
    team_run["always"], team_run["detail"] = True, "run_package is not available here"
    result = runner.invoke(app, ["pipelines", "run", "acme/triage", "--run-package"], env=TEST_ENV)
    assert result.exit_code == 1 and "Agree, and start the run?" not in result.output


def test_a_refusal_about_something_else_asks_nothing(team_run: Dict[str, Any]) -> None:
    team_run["detail"] = "The publish release: version 0.1.0 of acme/triage-tasks is taken"
    result = runner.invoke(app, ["pipelines", "run", "acme/triage"], env=TEST_ENV, input="y\n")
    assert result.exit_code == 1 and "is taken" in result.output
    assert "Agree, and start the run?" not in result.output and len(team_run["posts"]) == 1


def test_asking_keeps_json_output_to_the_run_alone(team_run: Dict[str, Any]) -> None:
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "-o", "json"], env=TEST_ENV, input="y\n"
    )
    assert result.exit_code == 0, result.output
    # The test runner echoes the typed answer to stdout; a terminal doesn't.
    assert json.loads(result.stdout.removeprefix(" y\n"))["id"] == "run1"
    assert "This team run would, as you" in result.stderr
    assert "Agree, and start the run?" in result.stderr


def test_run_help_says_what_the_agreement_is(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "run", "--help"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    # CI renders help in colour, which splits an option's name with escape codes.
    text = " ".join(strip_ansi(result.output).replace("│", " ").split())
    for expected in ("--launch", "--spend-limit", "--allow-host", "--run-package"):
        assert expected in text
    for gone in ("--sample", "--run-code", "--read-personal-traces"):
        assert gone not in text
    assert "can use your key and the secrets you pass" in text


def test_run_refuses_a_secret_missing_from_the_environment(api: Dict[str, Any]) -> None:
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--secret", "NOPE"], env=TEST_ENV
    )
    assert result.exit_code == 1 and "NOPE" in result.output and "post" not in api


def test_get_shows_counts_what_waits_and_outputs(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "get", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert api["get"] == "/pipelines/runs/run1"
    for text in (
        "Package: acme/triage@0.2.0",
        "scout",
        "Waiting on you at review (job batch): Keep these ideas?",
        "prime pipelines reply run1 --job batch --patch JSON",
        "report/report.md",
    ):
        assert text in result.output


def test_get_as_json_is_the_run_with_what_waits(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "get", "run1", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    found = json.loads(result.output)
    assert found["id"] == "run1"
    assert found["state"]["waiting_on_people"] == RUN["state"]["waiting_on_people"]


def test_get_shows_what_failed_and_what_the_run_started_and_spent(
    api: Dict[str, Any], monkeypatch
) -> None:
    run = {
        **RUN,
        "state": {
            "stages": {"grade": {"terminal": 2, "held": 1, "odd": 1}},
            "waiting_on_people": [],
            "failures": [
                {"job": "t3", "stage": "grade", "reason": "no reply", "error": "TimeoutError"}
            ],
        },
        "evaluations": [
            {"node": "grade", "evaluation_id": "ev1", "taskset": "acme/qa", "model": "m1"}
        ],
        "trainings": [{"node": "tune", "training_id": "tr1"}],
        "deployments": [{"node": "tune", "training_id": "tr1", "deployment_id": "dep1"}],
        "spend_limit": 20.0,
        "spent": 1.5,
    }
    monkeypatch.setattr(
        "prime_cli.client.APIClient.get", lambda self, endpoint, params=None, timeout=None: run
    )
    result = runner.invoke(app, ["pipelines", "get", "run1"], env={**TEST_ENV, "COLUMNS": "2000"})
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.replace("│", " ").replace("┃", " ").split())
    for expected in (
        "stage held done odd",
        "grade 1 2 1",
        "Failed at grade (job t3): TimeoutError no reply",
        "Evaluation ev1 (grade): acme/qa with m1",
        "Training tr1 (tune)",
        "Deployment dep1 (tune)",
        "Spend limit: $20.00, $1.50 spent",
    ):
        assert expected in text
    assert "Waiting on you" not in text


def test_stop_asks_first(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "stop", "run1"], env=TEST_ENV, input="n\n")
    assert result.exit_code == 0 and "Stop run run1?" in result.output and "post" not in api
    result = runner.invoke(app, ["pipelines", "stop", "run1"], env=TEST_ENV, input="y\n")
    assert result.exit_code == 0 and api["post"] == ("/pipelines/runs/run1/stop", None)


def test_stop_without_a_terminal_stops_nothing_unless_told_yes(api: Dict[str, Any]) -> None:
    # No answer to read: it ends, rather than waiting, and nothing is stopped.
    result = runner.invoke(app, ["pipelines", "--plain", "stop", "run1"], env=TEST_ENV, input="")
    assert result.exit_code == 1 and "post" not in api
    result = runner.invoke(app, ["pipelines", "--plain", "stop", "run1", "--yes"], env=TEST_ENV)
    assert result.exit_code == 0 and api["post"] == ("/pipelines/runs/run1/stop", None)


def test_the_group_is_eight_commands(api: Dict[str, Any]) -> None:
    from prime_cli.commands.pipelines import app as pipelines

    assert [command.name for command in pipelines.registered_commands] == [
        "run",
        "list",
        "runs",
        "get",
        "reply",
        "outputs",
        "stop",
        "delete",
    ]


@pytest.mark.parametrize("command", ["run", "get", "list", "runs"])
def test_help_documents_the_json_output(api: Dict[str, Any], command) -> None:
    result = runner.invoke(app, ["pipelines", command, "--help"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    text = " ".join(strip_ansi(result.output).replace("│", " ").split())
    assert "JSON output (--output json):" in text and "pipeline_id" in text
    assert "package?" in text and "sample" not in text
    assert (".state.waiting_on_people[]" in text) == (command == "get")
    assert (".total = number" in text) == (command in ("list", "runs"))


def test_get_shows_bracketed_text_as_written(api: Dict[str, Any], monkeypatch) -> None:
    run = {
        **RUN,
        "error": "bad [type=missing]",
        "state": {
            **RUN["state"],
            "waiting_on_people": [
                {**RUN["state"]["waiting_on_people"][0], "reason": "Choose [keep/drop]"}
            ],
        },
        "outputs": [{"name": "[/red].md", "node": "report", "bytes": 3}],
    }
    monkeypatch.setattr(
        "prime_cli.client.APIClient.get", lambda self, endpoint, params=None, timeout=None: run
    )
    result = runner.invoke(app, ["pipelines", "get", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "bad [type=missing]" in result.output and "Choose [keep/drop]" in result.output
    assert "report/[/red].md" in result.output


@pytest.mark.parametrize(
    ("node", "name"),
    [
        ("report", "../evil"),
        ("report", "D:evil"),
        ("report", ".."),
        # To Windows these are `..`: it drops a component's trailing dots and spaces.
        (".. ", "evil"),
        ("report", ".. ."),
        ("", "evil"),
    ],
)
def test_outputs_refuse_a_name_that_leaves_the_directory(monkeypatch, tmp_path, node, name) -> None:
    run = {**RUN, "outputs": [{"name": name, "node": node, "bytes": 1}]}
    monkeypatch.setattr(
        "prime_cli.client.APIClient.get", lambda self, endpoint, params=None, timeout=None: run
    )
    result = runner.invoke(
        app, ["pipelines", "outputs", "run1", "--dir", str(tmp_path)], env=TEST_ENV
    )
    assert result.exit_code == 1 and "refusing" in result.output
    assert not (tmp_path.parent / "evil").exists()


def test_errors_with_markup_print_as_written(api: Dict[str, Any]) -> None:
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--secret", "[/red]"], env=TEST_ENV
    )
    assert result.exit_code == 1 and "[/red]" in result.output


def test_outputs_never_write_through_a_planted_symlink(
    api: Dict[str, Any], monkeypatch, tmp_path
) -> None:
    from contextlib import contextmanager

    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            yield b"# R"

    @contextmanager
    def stream(self: Any, method: str, url: str, **kwargs: Any):
        yield Response()

    monkeypatch.setattr("httpx.Client.stream", stream)
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (tmp_path / "out" / "report").mkdir(parents=True)
    (tmp_path / "out" / "report" / ".report.md.part").symlink_to(victim)
    result = runner.invoke(
        app, ["pipelines", "outputs", "run1", "--dir", str(tmp_path / "out")], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert victim.read_text() == "keep"
    assert (tmp_path / "out" / "report" / "report.md").read_text() == "# R"


def test_outputs_encode_names_in_the_download_url(
    api: Dict[str, Any], monkeypatch, tmp_path
) -> None:
    from contextlib import contextmanager

    run = {**RUN, "outputs": [{"name": "report?v=1#x.md", "node": "report", "bytes": 3}]}
    monkeypatch.setattr(
        "prime_cli.client.APIClient.get", lambda self, endpoint, params=None, timeout=None: run
    )
    urls = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            yield b"# R"

    @contextmanager
    def stream(self: Any, method: str, url: str, **kwargs: Any):
        urls.append(url)
        yield Response()

    monkeypatch.setattr("httpx.Client.stream", stream)
    result = runner.invoke(
        app, ["pipelines", "outputs", "run1", "--dir", str(tmp_path)], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert urls[0].endswith("/api/v1/pipelines/runs/run1/outputs/report/report%3Fv%3D1%23x.md")


def test_outputs_of_the_same_name_in_two_folders_are_both_saved(
    api: Dict[str, Any], monkeypatch, tmp_path
) -> None:
    from contextlib import contextmanager

    run = {
        **RUN,
        "outputs": [
            {"name": "report.md", "node": "a", "bytes": 1},
            {"name": "report.md", "node": "b", "bytes": 1},
        ],
    }
    monkeypatch.setattr(
        "prime_cli.client.APIClient.get", lambda self, endpoint, params=None, timeout=None: run
    )

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            yield self.body

    @contextmanager
    def stream(self: Any, method: str, url: str, **kwargs: Any):
        yield Response(url.rsplit("/", 2)[1].encode())

    monkeypatch.setattr("httpx.Client.stream", stream)
    result = runner.invoke(
        app, ["pipelines", "outputs", "run1", "--dir", str(tmp_path)], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "a" / "report.md").read_text() == "a"
    assert (tmp_path / "b" / "report.md").read_text() == "b"


WAITING = [
    {"job": "review.t1", "stage": "review", "reason": "3 ideas: keep or drop each", "by": "review"},
    {"job": "batch", "stage": "gate", "reason": "Ship it?", "by": "gate"},
    {"job": "t9", "stage": "gate", "reason": "", "by": "operator"},
]


@pytest.fixture
def review(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    calls: Dict[str, Any] = {"posts": []}
    run = {**RUN, "state": {**RUN["state"], "waiting_on_people": WAITING}}

    def post(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None, timeout: Any = None
    ) -> Dict[str, Any]:
        calls["posts"].append((endpoint, json))
        return {"seq": 7}

    def get(self: Any, endpoint: str, params: Any = None, timeout: Any = None) -> Dict[str, Any]:
        calls["get"] = endpoint
        return calls.get("run", run)

    monkeypatch.setattr("prime_cli.client.APIClient.post", post)
    monkeypatch.setattr("prime_cli.client.APIClient.get", get)
    return calls


def _reply(review: Dict[str, Any], *args: str) -> Any:
    result = runner.invoke(app, ["pipelines", "reply", "run1", *args], env=TEST_ENV)
    sent = [body["control"] for endpoint, body in review["posts"]]
    assert all(endpoint == "/pipelines/runs/run1/control" for endpoint, _ in review["posts"])
    return result, sent


def test_get_lists_what_waits_and_how_to_answer_each(review: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "get", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert review["get"] == "/pipelines/runs/run1"
    for text in (
        "Waiting on you at review (job review.t1): 3 ideas: keep or drop each",
        "Waiting on you at gate (job batch): Ship it?",
        "Waiting on you at gate (job t9): no reason given",
        "prime pipelines reply run1 --job review.t1 --patch JSON",
    ):
        assert text in result.output
    assert review["posts"] == []


def test_get_says_why_a_reply_was_not_applied(review: Dict[str, Any]) -> None:
    review["run"] = {
        **RUN,
        "state": {
            "waiting_on_people": [],
            "skipped_controls": [{"seq": 3, "error": "KeyError: [job]"}],
        },
    }
    result = runner.invoke(app, ["pipelines", "get", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "Not applied (control 3): KeyError: [job]" in result.output
    assert "Waiting on you" not in result.output


def test_reply_sends_the_patch_as_an_answer_to_the_job(review: Dict[str, Any]) -> None:
    patch = '{"approved": true, "note": "fine"}'
    result, sent = _reply(review, "--job", "batch", "--patch", patch)
    assert result.exit_code == 0, result.output
    assert sent == [{"type": "answer", "job": "batch", "patch": {"approved": True, "note": "fine"}}]
    assert "Sent (control 7)" in result.output
    assert "prime pipelines get run1 says why" in result.output


def test_reply_needs_no_job_while_only_one_waits(review: Dict[str, Any]) -> None:
    review["run"] = {**RUN, "state": {"waiting_on_people": WAITING[:1]}}
    result, sent = _reply(review, "--patch", "{}")
    assert result.exit_code == 0, result.output
    assert sent == [{"type": "answer", "job": "review.t1", "patch": {}}]


@pytest.mark.parametrize(
    "args, error",
    [
        (["--patch", "{}"], "3 jobs wait on you"),
        (["--job", "nope", "--patch", "{}"], "nothing waiting matches"),
        (["--job", "batch", "--patch", "{"], "--patch is not JSON"),
        (["--job", "batch", "--patch", "[1]"], "--patch is a JSON object"),
        (["--job", "batch", "--patch", '"yes"'], "--patch is a JSON object"),
    ],
)
def test_a_reply_that_cannot_be_sent_sends_nothing(review: Dict[str, Any], args, error) -> None:
    result, sent = _reply(review, *args)
    assert result.exit_code == 1 and error in result.output and sent == []


def test_reply_without_a_patch_sends_nothing(review: Dict[str, Any]) -> None:
    result, sent = _reply(review, "--job", "batch")
    assert result.exit_code == 2 and sent == []


def test_a_refused_reply_says_why(review: Dict[str, Any], monkeypatch) -> None:
    from prime_cli.client import APIError

    def post(self: Any, endpoint: str, json: Any = None, timeout: Any = None) -> Any:
        raise APIError("HTTP 409: This run has finished")

    monkeypatch.setattr("prime_cli.client.APIClient.post", post)
    result = runner.invoke(
        app, ["pipelines", "reply", "run1", "--job", "batch", "--patch", "{}"], env=TEST_ENV
    )
    assert result.exit_code == 1 and "This run has finished" in result.output


def test_stop_with_drain_lets_running_work_finish(review: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "stop", "run1", "--drain"], env=TEST_ENV, input="n\n")
    assert result.exit_code == 0 and "Wind down run run1?" in result.output
    assert review["posts"] == []
    result = runner.invoke(app, ["pipelines", "stop", "run1", "--drain", "-y"], env=TEST_ENV)
    assert result.exit_code == 0 and "Sent (control 7)" in result.output
    assert review["posts"] == [("/pipelines/runs/run1/control", {"control": {"type": "drain"}})]
    assert "get" not in review


@pytest.mark.parametrize("gone", ["status", "cards", "drain", "refresh"])
def test_the_commands_folded_away_are_gone(review: Dict[str, Any], gone) -> None:
    result = runner.invoke(app, ["pipelines", gone, "run1"], env=TEST_ENV)
    assert result.exit_code == 2 and review["posts"] == [] and "get" not in review


def test_control_paths_encode_the_run_id(review: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "stop", "r/1", "--drain", "--yes"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert review["posts"][0][0] == "/pipelines/runs/r%2F1/control"


def test_outputs_say_when_the_folder_cannot_be_made(api: Dict[str, Any], tmp_path) -> None:
    (tmp_path / "file").write_text("in the way")
    result = runner.invoke(
        app, ["pipelines", "outputs", "run1", "--dir", str(tmp_path / "file" / "out")], env=TEST_ENV
    )
    assert result.exit_code == 1 and "downloading report.md" in result.output
    assert "Traceback" not in result.output


def test_an_interrupted_download_leaves_no_partial_file(
    api: Dict[str, Any], monkeypatch, tmp_path
) -> None:
    from contextlib import contextmanager

    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            yield b"# R"
            raise KeyboardInterrupt

    @contextmanager
    def stream(self: Any, method: str, url: str, **kwargs: Any):
        yield Response()

    monkeypatch.setattr("httpx.Client.stream", stream)
    result = runner.invoke(
        app, ["pipelines", "outputs", "run1", "--dir", str(tmp_path)], env=TEST_ENV
    )
    assert result.exit_code != 0
    assert list((tmp_path / "report").iterdir()) == []


def test_reply_quotes_the_names_it_suggests_when_several_match(
    review: Dict[str, Any],
) -> None:
    waiting = [
        {"job": "a; rm -rf ~", "stage": "ask", "reason": "Which?"},
        {"job": "b", "stage": "ask", "reason": "Which?"},
    ]
    review["run"] = {**RUN, "state": {"waiting_on_people": waiting}}
    env = {**TEST_ENV, "COLUMNS": "2000"}
    result = runner.invoke(app, ["pipelines", "reply", "run1", "--patch", "{}"], env=env)
    assert result.exit_code == 1 and review["posts"] == []
    assert "--job 'a; rm -rf ~', --job b" in result.output


PIPELINES = {
    "pipelines": [
        {
            "id": "p1",
            "name": "nightly [draft]",
            "team_id": "t1",
            "version": "a" * 64,
            "package": "acme/triage",
            "updated_at": "2026-11-11T10:20:30Z",
            "runs": 2,
            "latest_run": {**RUN, "state": None},
        },
        {
            "id": "p2",
            "name": "idle",
            "team_id": "t1",
            "version": "b" * 64,
            "package": "acme/idle@1.0",
            "updated_at": "2026-11-10T08:00:00Z",
            "runs": 0,
            "latest_run": None,
        },
    ],
    "total": 2,
    "page": 1,
    "per_page": 20,
}
RUNS = {
    "runs": [
        {**RUN, "state": None, "started_at": "2026-11-11T10:21:00Z"},
        {**RUN, "id": "run0", "status": "FAILED", "stopped": True, "state": None},
    ],
    "total": 41,
    "page": 1,
    "per_page": 2,
}


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """The platform's routes for saved pipelines: every request, and what each answers."""
    calls: Dict[str, Any] = {"requests": [], "refuse": None}

    def request(
        self: Any,
        method: str,
        endpoint: str,
        params: Any = None,
        json: Any = None,
        timeout: Any = None,
    ) -> Dict[str, Any]:
        calls["requests"].append((method, endpoint, params or json))
        if calls["refuse"] and method == calls["refuse"][0]:
            raise calls["refuse"][1]
        if method == "DELETE":
            return {"id": "p1", "name": "nightly [draft]", "runs": 2}
        if endpoint == "/pipelines":
            return calls.get("pipelines", PIPELINES)
        if endpoint == "/pipelines/runs":
            return RUN if method == "POST" else calls.get("runs", RUNS)
        return PIPELINES["pipelines"][0]

    monkeypatch.setattr("prime_cli.client.APIClient.request", request)
    # The team picked with `prime switch`: none, unless a test sets one.
    monkeypatch.setattr(
        "prime_cli.commands.pipelines.Config",
        lambda: type("Picked", (), {"team_id": calls.get("team")})(),
    )
    return calls


def test_list_shows_each_pipeline_with_its_latest_run(saved: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "list", "-t", "t1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert saved["requests"] == [
        ("GET", "/pipelines", {"team_id": "t1", "page": 1, "per_page": 20})
    ]
    for text in (
        "p1",
        "nightly [draft]",
        "acme/triage",
        "acme/idle@1.0",
        "run1",
        "RUNNING",
        "2026-11-11 10:20",
    ):
        assert text in result.output
    assert "Total: 2 pipelines" in result.output


def test_list_pages_and_prints_json_as_the_platform_sent_it(saved: Dict[str, Any]) -> None:
    result = runner.invoke(
        app, ["pipelines", "list", "-n", "5", "-p", "3", "-o", "json"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    # No team picked: the caller's own pipelines.
    assert saved["requests"] == [("GET", "/pipelines", {"page": 3, "per_page": 5})]
    assert json.loads(result.output) == PIPELINES


def test_list_says_when_there_is_nothing(saved: Dict[str, Any]) -> None:
    saved["pipelines"] = {"pipelines": [], "total": 0, "page": 1, "per_page": 20}
    result = runner.invoke(app, ["pipelines", "list"], env=TEST_ENV)
    assert result.exit_code == 0 and "No pipelines yet." in result.output
    result = runner.invoke(app, ["pipelines", "list", "--page", "2"], env=TEST_ENV)
    assert result.exit_code == 0 and "No more pipelines." in result.output


@pytest.mark.parametrize("command", ["list", "runs"])
@pytest.mark.parametrize("paging", [["--num", "0"], ["--num", "101"], ["--page", "0"]])
def test_a_page_the_platform_would_refuse_is_refused_first(
    saved: Dict[str, Any], command, paging
) -> None:
    result = runner.invoke(app, ["pipelines", command, *paging], env=TEST_ENV)
    assert result.exit_code == 1 and saved["requests"] == []
    assert "--num must be from 1 to 100, and --page at least 1" in result.output


@pytest.mark.parametrize("command", ["list", "runs"])
def test_the_largest_page_is_asked_for_as_given(saved: Dict[str, Any], command) -> None:
    result = runner.invoke(app, ["pipelines", command, "--num", "100"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert saved["requests"][0][2]["per_page"] == 100


def test_runs_lists_one_pipelines_or_the_accounts(saved: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "runs", "-t", "t1", "-n", "2"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert saved["requests"] == [
        ("GET", "/pipelines/runs", {"team_id": "t1", "page": 1, "per_page": 2})
    ]
    for text in (
        "run1",
        "acme/triage@0.2.0",
        "RUNNING",
        "2026-11-11 10:21",
        "run0",
        "STOPPED",
        "aaaaaaaaaaaa",
    ):
        assert text in result.output
    assert "2 of 41 runs. Use --page 2 to see more." in result.output

    # A pipeline has its own account: no team is sent with it.
    saved["requests"].clear()
    result = runner.invoke(app, ["pipelines", "runs", "p1", "-t", "t1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert saved["requests"] == [
        ("GET", "/pipelines/runs", {"pipeline_id": "p1", "page": 1, "per_page": 20})
    ]
    # Without --team, the team picked with `prime switch`.
    saved["requests"].clear()
    saved["team"] = "picked"
    result = runner.invoke(app, ["pipelines", "runs", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0 and json.loads(result.output) == RUNS
    assert saved["requests"][0][2]["team_id"] == "picked"


def test_run_of_a_pipeline_id_starts_its_latest_version(saved: Dict[str, Any]) -> None:
    env = {**TEST_ENV, "GITHUB_TOKEN": "gh"}
    args = ["--secret", "GITHUB_TOKEN", "--run-package", "--allow-host", "api.github.com"]
    result = runner.invoke(app, ["pipelines", "run", "p1", *args, "-t", "t1"], env=env)
    assert result.exit_code == 0, result.output
    ((method, endpoint, body),) = saved["requests"]
    assert (method, endpoint) == ("POST", "/pipelines/runs")
    # No package, and no team: the pipeline runs the launch it was saved with, as its own account.
    assert body == {
        "pipeline_id": "p1",
        "secrets": {"GITHUB_TOKEN": "gh"},
        "network": {"allow": ["api.github.com"]},
        "run_package": True,
    }
    assert "prime pipelines get run1" in result.output


def test_run_saves_a_new_version_of_a_pipeline_under_the_name_it_has(
    saved: Dict[str, Any],
) -> None:
    from prime_cli.client import APIError

    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--pipeline", "p/1"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    # Read first: a save names the pipeline, and it keeps the name it has.
    assert [request[:2] for request in saved["requests"]] == [
        ("GET", "/pipelines/p%2F1"),
        ("POST", "/pipelines/runs"),
    ]
    body = saved["requests"][1][2]
    assert (body["pipeline_id"], body["name"]) == ("p/1", "nightly [draft]")

    saved["requests"].clear()
    args = ["pipelines", "run", "acme/triage", "--pipeline", "p1", "--name", "weekly"]
    result = runner.invoke(app, args, env=TEST_ENV)
    assert result.exit_code == 0, result.output
    ((method, _, body),) = saved["requests"]
    assert method == "POST" and (body["pipeline_id"], body["name"]) == ("p1", "weekly")

    saved["requests"].clear()
    saved["refuse"] = ("GET", APIError("HTTP 404: Pipeline not found"))
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--pipeline", "p9"], env=TEST_ENV
    )
    assert result.exit_code == 1 and "Pipeline not found" in result.output
    assert [request[0] for request in saved["requests"]] == ["GET"]


def test_run_tells_a_package_from_a_pipeline_id(saved: Dict[str, Any], tmp_path) -> None:
    saved["team"] = "picked"
    result = runner.invoke(app, ["pipelines", "run", "acme/p1@latest"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    body = saved["requests"][0][2]
    assert body["package"] == "acme/p1@latest" and body["pipeline_id"] is None
    # Without --team, the team picked with `prime switch`.
    assert body["team_id"] == "picked"

    saved["requests"].clear()
    # A file is neither: the launch file goes with --launch, after the package.
    file = tmp_path / "launch.toml"
    file.write_text("")
    for target in (str(file), "a/b/c", "acme/", "acme/p1@", "two words"):
        neither = runner.invoke(app, ["pipelines", "run", target], env=TEST_ENV)
        assert neither.exit_code == 1 and "neither a package" in neither.output, target
    options = (
        ["--pipeline", "p2"],
        ["--launch", str(file)],
        ["--spend-limit", "5"],
        ["--name", "nightly"],
    )
    for option in options:
        both = runner.invoke(app, ["pipelines", "run", "p1", *option], env=TEST_ENV)
        assert both.exit_code == 1 and "go with a package" in both.output
    assert saved["requests"] == []


def test_run_says_how_to_choose_between_pipelines_of_one_name(saved: Dict[str, Any]) -> None:
    from prime_cli.client import APIError

    detail = (
        "Several pipelines are named 'triage': p1, p2. Name the one to add this version to "
        "with pipeline_id."
    )
    saved["refuse"] = ("POST", APIError(f"HTTP 409: {detail}"))
    env = {**TEST_ENV, "COLUMNS": "2000"}
    result = runner.invoke(app, ["pipelines", "run", "acme/triage"], env=env)
    assert result.exit_code == 1 and "p1, p2" in result.output
    assert "Pass one of these ids with --pipeline." in result.output


def test_delete_asks_first_naming_the_pipeline_and_its_runs(saved: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "delete", "p1"], env=TEST_ENV, input="n\n")
    assert result.exit_code == 0, result.output
    assert 'Delete pipeline "nightly [draft]" (p1)?' in result.output
    assert "Its 2 runs and their outputs are deleted with it." in result.output
    assert "Cancelled." in result.output
    assert saved["requests"] == [("GET", "/pipelines/p1", None)]

    result = runner.invoke(app, ["pipelines", "delete", "p1"], env=TEST_ENV, input="y\n")
    assert result.exit_code == 0, result.output
    assert saved["requests"][-1] == ("DELETE", "/pipelines/p1", None)
    assert "Deleted pipeline nightly [draft] (p1) with 2 runs." in result.output


def test_delete_with_yes_asks_nothing_and_encodes_the_id(saved: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "--plain", "delete", "p/1", "--yes"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert [request[:2] for request in saved["requests"]] == [
        ("GET", "/pipelines/p%2F1"),
        ("DELETE", "/pipelines/p%2F1"),
    ]
    assert "[y/N]" not in result.output


def test_delete_without_an_answer_deletes_nothing(saved: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "delete", "p1"], env=TEST_ENV, input="")
    assert result.exit_code == 1
    assert [request[0] for request in saved["requests"]] == ["GET"]


def test_delete_refused_for_a_live_run_says_how_to_stop_it(saved: Dict[str, Any]) -> None:
    from prime_cli.client import APIError

    saved["refuse"] = (
        "DELETE",
        APIError("HTTP 409: Run run9 of this pipeline is still running. Stop it first."),
    )
    env = {**TEST_ENV, "COLUMNS": "2000"}
    result = runner.invoke(app, ["pipelines", "delete", "p1", "-y"], env=env)
    assert result.exit_code == 1
    assert "Run run9 of this pipeline is still running." in result.output
    assert "Stop it with: prime pipelines stop run9" in result.output


def test_delete_of_a_pipeline_that_is_not_yours_is_not_found(saved: Dict[str, Any]) -> None:
    from prime_cli.client import APIError

    saved["refuse"] = ("GET", APIError("HTTP 404: Pipeline not found"))
    result = runner.invoke(app, ["pipelines", "delete", "p1", "-y"], env=TEST_ENV)
    assert result.exit_code == 1 and "Pipeline not found" in result.output
    assert [request[0] for request in saved["requests"]] == ["GET"]


def test_run_reads_the_launch_file_as_utf8_whatever_the_locale(
    api: Dict[str, Any], monkeypatch, tmp_path
) -> None:
    file = tmp_path / "launch.json"
    text = json.dumps({"config": {"name": "café ✓"}}, ensure_ascii=False)
    file.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))  # with a BOM, as Notepad saves it
    monkeypatch.setattr("locale.getpreferredencoding", lambda *args: "cp1252")
    monkeypatch.setattr("locale.getencoding", lambda: "cp1252", raising=False)
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--launch", str(file)], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert api["post"][1]["config"] == {"name": "café ✓"}


def test_run_says_when_the_launch_file_is_not_utf8(api: Dict[str, Any], tmp_path) -> None:
    file = tmp_path / "launch.toml"
    file.write_bytes(b'[config]\nname = "caf\xe9"\n')
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--launch", str(file)], env=TEST_ENV
    )
    assert result.exit_code == 1 and "is not UTF-8 text" in result.output
    assert "Traceback" not in result.output and "post" not in api


def test_run_says_when_the_launch_file_cannot_be_read(
    api: Dict[str, Any], monkeypatch, tmp_path
) -> None:
    file = tmp_path / "launch.toml"
    file.write_text("")

    def read_text(self: Any, *args: Any, **kwargs: Any) -> str:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("pathlib.Path.read_text", read_text)
    result = runner.invoke(
        app, ["pipelines", "run", "acme/triage", "--launch", str(file)], env=TEST_ENV
    )
    assert result.exit_code == 1 and "can't read" in result.output
    assert "Traceback" not in result.output and "post" not in api
