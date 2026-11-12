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
    "status": "RUNNING",
    "error": None,
    "stopped": False,
    "sample": None,
    "outputs": [{"name": "report.md", "node": "report", "bytes": 3}],
    "state": {
        "nodes": {"scout": {"done": 1, "running": 0, "held": 0, "failed": 0, "cost": 0.25}},
        "cards": [{"job": "graph", "node": "review", "question": "Keep or drop each idea."}],
        "cost": {"usd": 0.25},
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


def test_run_submits_the_graph_with_the_secrets_it_names(api: Dict[str, Any], tmp_path) -> None:
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps({"name": "g", "nodes": {}}))
    env = {**TEST_ENV, "GITHUB_TOKEN": "gh"}
    result = runner.invoke(
        app,
        ["pipelines", "run", str(graph), "--secret", "GITHUB_TOKEN", "--sample", "2"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    endpoint, body = api["post"]
    assert endpoint == "/pipelines/runs"
    assert body["graph"] == {"name": "g", "nodes": {}}
    assert body["secrets"] == {"GITHUB_TOKEN": "gh"} and body["sample"] == 2
    assert body["pipeline_id"] is None
    assert "prime pipelines get run1" in result.output


def test_run_saves_a_new_version_of_a_named_pipeline(api: Dict[str, Any], tmp_path) -> None:
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps({"name": "g", "nodes": {}}))
    result = runner.invoke(
        app, ["pipelines", "run", str(graph), "--pipeline", "p1", "-t", "t1"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert api["post"][1]["pipeline_id"] == "p1" and api["post"][1]["team_id"] == "t1"
    assert "Pipeline: p1" in result.output


def test_run_sends_the_hosts_it_adds_and_what_its_starter_agreed_to(
    api: Dict[str, Any], tmp_path
) -> None:
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps({"name": "g", "nodes": {}}))
    result = runner.invoke(app, ["pipelines", "run", str(graph)], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    body = api["post"][1]
    assert body["network"] == {"allow": []}
    assert (body["read_personal_traces"], body["run_code"]) == (False, False)

    hosts = ["--allow-host", "api.github.com", "--allow-host", "*.example.com"]
    result = runner.invoke(
        app,
        ["pipelines", "run", str(graph), *hosts, "--read-personal-traces", "--run-code"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    body = api["post"][1]
    assert body["network"] == {"allow": ["api.github.com", "*.example.com"]}
    assert (body["read_personal_traces"], body["run_code"]) == (True, True)


@pytest.mark.parametrize(
    ("detail", "passed", "hint"),
    [
        (
            "The Code nodes ['glue'] run code saved in this team's pipeline on the run's "
            "machine. Read their source, then start the run with run_code to agree.",
            [],
            "To agree, pass --run-code.",
        ),
        (
            "The Traces inputs ['mine'] read your personal traces into outputs this team can "
            "see. Start the run with read_personal_traces to agree.",
            ["--run-code"],
            "To agree, pass --read-personal-traces.",
        ),
        ("No Code node of the graph names the secrets ['X']", [], None),
        # Already agreed to: the refusal is about something else, so no option is suggested.
        ("run_code is not available here", ["--run-code"], None),
    ],
)
def test_run_names_the_option_for_an_agreement_the_platform_asks_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path, detail, passed, hint
) -> None:
    from prime_cli.client import APIError

    def post(self: Any, endpoint: str, json: Any = None, timeout: Any = None) -> Dict[str, Any]:
        raise APIError(f"HTTP 422: {detail}")

    monkeypatch.setattr("prime_cli.client.APIClient.post", post)
    graph = tmp_path / "g.json"
    graph.write_text("{}")
    env = {**TEST_ENV, "COLUMNS": "2000"}
    result = runner.invoke(app, ["pipelines", "run", str(graph), *passed], env=env)
    assert result.exit_code == 1 and detail in result.output
    assert ("To agree, pass" in result.output) == (hint is not None)
    assert hint is None or hint in result.output


def test_run_help_says_what_each_agreement_is(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "run", "--help"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    # CI renders help in colour, which splits an option's name with escape codes.
    text = " ".join(strip_ansi(result.output).replace("│", " ").split())
    for expected in ("--allow-host", "--read-personal-traces", "--run-code"):
        assert expected in text
    assert "can use your key and the secrets you pass" in text


def test_run_refuses_a_secret_missing_from_the_environment(api: Dict[str, Any], tmp_path) -> None:
    graph = tmp_path / "g.json"
    graph.write_text("{}")
    result = runner.invoke(app, ["pipelines", "run", str(graph), "--secret", "NOPE"], env=TEST_ENV)
    assert result.exit_code == 1 and "NOPE" in result.output and "post" not in api


def test_get_shows_counts_what_waits_cost_and_outputs(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "get", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert api["get"] == "/pipelines/runs/run1"
    for text in ("scout", "Waiting on you", "Keep or drop each idea.", "$0.25", "report/report.md"):
        assert text in result.output


def test_get_as_json_is_the_run_with_what_waits(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "get", "run1", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    found = json.loads(result.output)
    assert found["id"] == "run1" and found["state"]["cards"] == RUN["state"]["cards"]


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
    assert (".state.cards[]" in text) == (command == "get")
    assert (".total = number" in text) == (command in ("list", "runs"))


def test_get_shows_bracketed_text_as_written(api: Dict[str, Any], monkeypatch) -> None:
    run = {
        **RUN,
        "error": "bad [type=missing]",
        "state": {
            **RUN["state"],
            "cards": [{**RUN["state"]["cards"][0], "question": "Choose [keep/drop]"}],
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


def test_errors_with_markup_print_as_written(api: Dict[str, Any], tmp_path) -> None:
    graph = tmp_path / "g.json"
    graph.write_text("{}")
    result = runner.invoke(
        app, ["pipelines", "run", str(graph), "--secret", "[/red]"], env=TEST_ENV
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


def test_outputs_of_the_same_name_from_two_heads_are_both_saved(
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


CARDS = [
    {
        "job": "review.t1",
        "node": "review",
        "question": "Keep or drop each idea.",
        "items": ["a", "b", "c"],
        "reason": "3 ideas",
    },
    {"job": "graph", "node": "gate", "question": "Ship it? (yes | no)", "options": ["yes", "no"]},
    {"job": "graph", "node": "ask", "question": "Which repo?"},
]


@pytest.fixture
def review(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    calls: Dict[str, Any] = {"posts": []}
    run = {**RUN, "state": {**RUN["state"], "cards": CARDS}}

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
        "Waiting on you at review (job review.t1), review: Keep or drop each idea.",
        "3 ideas",
        "- b",
        "gate (job graph), choice",
        "Options: yes, no",
        "ask (job graph), question: Which repo?",
        "prime pipelines reply run1 --node gate --job graph --choice OPTION",
        "--note TEXT",
    ):
        assert text in result.output
    assert review["posts"] == []


def test_get_says_why_a_reply_was_not_applied(review: Dict[str, Any]) -> None:
    review["run"] = {
        **RUN,
        "state": {"cards": [], "skipped_controls": [{"seq": 3, "error": "KeyError: [job]"}]},
    }
    result = runner.invoke(app, ["pipelines", "get", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "Not applied (control 3): KeyError: [job]" in result.output
    assert "Waiting on you" not in result.output


@pytest.mark.parametrize(
    "args, keep",
    [
        (["--keep", "c", "--keep", "a"], ["a", "c"]),
        (["--drop", "b"], ["a", "c"]),
        (["--drop", "a", "--drop", "b", "--drop", "c"], []),
        (["--keep-all"], ["a", "b", "c"]),
    ],
)
def test_reply_to_a_review_sends_the_items_kept(review: Dict[str, Any], args, keep) -> None:
    result, sent = _reply(review, "--node", "review", *args, "--note", "fine")
    assert result.exit_code == 0, result.output
    assert sent == [
        {
            "type": "reply",
            "job": "review.t1",
            "node": "review",
            "reply": {"note": "fine", "keep": keep},
        }
    ]
    assert f"Sent (control 7), keeping {len(keep)} of 3" in result.output
    assert "prime pipelines get run1 says why" in result.output


def test_reply_with_a_choice_and_to_a_question(review: Dict[str, Any]) -> None:
    result, sent = _reply(review, "--node", "gate", "--choice", "no")
    assert result.exit_code == 0, result.output
    result, sent = _reply(review, "--node", "ask", "--job", "graph", "--note", "prime")
    assert result.exit_code == 0, result.output
    assert [control["reply"] for control in sent] == [
        {"note": "", "choice": "no"},
        {"note": "prime"},
    ]
    assert [(c["job"], c["node"]) for c in sent] == [("graph", "gate"), ("graph", "ask")]


@pytest.mark.parametrize(
    "args, error",
    [
        ([], "3 questions match"),
        (["--job", "graph", "--choice", "yes"], "2 questions match"),
        (["--node", "nope"], "nothing waiting matches; see prime pipelines get run1"),
        (["--node", "gate", "--choice", "maybe"], "choose one of: yes, no"),
        (["--node", "gate", "--keep", "a"], "it has no items"),
        (["--node", "review", "--choice", "yes"], "no options"),
        (["--node", "review"], "one of --keep, --drop or --keep-all"),
        (["--node", "review", "--keep", "a", "--drop", "b"], "one of --keep"),
        (["--node", "review", "--keep", "z"], "not an item of this review: z"),
        (["--node", "ask", "--note", "  "], "answer the question with --note"),
    ],
)
def test_a_reply_that_doesnt_fit_its_card_sends_nothing(
    review: Dict[str, Any], args, error
) -> None:
    result, sent = _reply(review, *args)
    assert result.exit_code == 1 and error in result.output and sent == []


def test_a_refused_reply_says_why(review: Dict[str, Any], monkeypatch) -> None:
    from prime_cli.client import APIError

    def post(self: Any, endpoint: str, json: Any = None, timeout: Any = None) -> Any:
        raise APIError("HTTP 409: This run has finished")

    monkeypatch.setattr("prime_cli.client.APIClient.post", post)
    result = runner.invoke(
        app, ["pipelines", "reply", "run1", "--node", "gate", "--choice", "yes"], env=TEST_ENV
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
    cards = [
        {"job": "a; rm -rf ~", "node": "ask", "question": "Which?"},
        {"job": "b", "node": "ask", "question": "Which?"},
    ]
    review["run"] = {**RUN, "state": {"cards": cards}}
    env = {**TEST_ENV, "COLUMNS": "2000"}
    result = runner.invoke(app, ["pipelines", "reply", "run1", "--note", "x"], env=env)
    assert result.exit_code == 1 and review["posts"] == []
    assert "--node ask --job 'a; rm -rf ~', --node ask --job b" in result.output


PIPELINES = {
    "pipelines": [
        {
            "id": "p1",
            "name": "nightly [draft]",
            "team_id": "t1",
            "version": "a" * 64,
            "updated_at": "2026-11-11T10:20:30Z",
            "runs": 2,
            "latest_run": {**RUN, "state": None},
        },
        {
            "id": "p2",
            "name": "idle",
            "team_id": "t1",
            "version": "b" * 64,
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
    for text in ("p1", "nightly [draft]", "run1", "RUNNING", "2026-11-11 10:20", "idle"):
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
    result = runner.invoke(app, ["pipelines", "list", "--num", "0"], env=TEST_ENV)
    assert result.exit_code == 1 and "at least 1" in result.output


def test_runs_lists_one_pipelines_or_the_accounts(saved: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "runs", "-t", "t1", "-n", "2"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert saved["requests"] == [
        ("GET", "/pipelines/runs", {"team_id": "t1", "page": 1, "per_page": 2})
    ]
    for text in ("run1", "RUNNING", "2026-11-11 10:21", "run0", "STOPPED", "aaaaaaaaaaaa"):
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
    args = ["--secret", "GITHUB_TOKEN", "--run-code", "--allow-host", "api.github.com"]
    result = runner.invoke(app, ["pipelines", "run", "p1", *args, "-t", "t1"], env=env)
    assert result.exit_code == 0, result.output
    ((method, endpoint, body),) = saved["requests"]
    assert (method, endpoint) == ("POST", "/pipelines/runs")
    # No graph, and no team: the pipeline runs as its own account.
    assert body == {
        "pipeline_id": "p1",
        "sample": None,
        "secrets": {"GITHUB_TOKEN": "gh"},
        "network": {"allow": ["api.github.com"]},
        "read_personal_traces": False,
        "run_code": True,
    }
    assert "prime pipelines get run1" in result.output


def test_run_tells_a_file_from_a_pipeline_id(saved: Dict[str, Any], tmp_path) -> None:
    # A file that exists is a graph, whatever it is called.
    graph = tmp_path / "p1"
    graph.write_text(json.dumps({"name": "g", "nodes": {}}))
    result = runner.invoke(app, ["pipelines", "run", str(graph)], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert saved["requests"][0][2]["graph"] == {"name": "g", "nodes": {}}
    assert saved["requests"][0][2]["pipeline_id"] is None

    saved["requests"].clear()
    missing = runner.invoke(app, ["pipelines", "run", "missing.json"], env=TEST_ENV)
    assert missing.exit_code == 1 and "neither a file nor a pipeline's id" in missing.output
    both = runner.invoke(app, ["pipelines", "run", "p1", "--pipeline", "p2"], env=TEST_ENV)
    assert both.exit_code == 1 and "--pipeline goes with a graph file" in both.output
    assert saved["requests"] == []


def test_run_says_how_to_choose_between_pipelines_of_one_name(
    saved: Dict[str, Any], tmp_path
) -> None:
    from prime_cli.client import APIError

    detail = (
        "Several pipelines are named 'g': p1, p2. Name the one to add this version to "
        "with pipeline_id."
    )
    saved["refuse"] = ("POST", APIError(f"HTTP 409: {detail}"))
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps({"name": "g", "nodes": {}}))
    env = {**TEST_ENV, "COLUMNS": "2000"}
    result = runner.invoke(app, ["pipelines", "run", str(graph)], env=env)
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


def test_run_says_when_the_graph_file_cannot_be_read(
    api: Dict[str, Any], monkeypatch, tmp_path
) -> None:
    graph = tmp_path / "g.json"
    graph.write_text("{}")

    def read_text(self: Any, *args: Any, **kwargs: Any) -> str:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("pathlib.Path.read_text", read_text)
    result = runner.invoke(app, ["pipelines", "run", str(graph)], env=TEST_ENV)
    assert result.exit_code == 1 and "can't read" in result.output
    assert "Traceback" not in result.output and "post" not in api
