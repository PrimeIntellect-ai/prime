import json
from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
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
    assert "prime pipelines status run1" in result.output


def test_run_saves_a_new_version_of_a_named_pipeline(api: Dict[str, Any], tmp_path) -> None:
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps({"name": "g", "nodes": {}}))
    result = runner.invoke(
        app, ["pipelines", "run", str(graph), "--pipeline-id", "p1"], env=TEST_ENV
    )
    assert result.exit_code == 0, result.output
    assert api["post"][1]["pipeline_id"] == "p1"
    assert "Pipeline: p1" in result.output


def test_run_refuses_a_secret_missing_from_the_environment(api: Dict[str, Any], tmp_path) -> None:
    graph = tmp_path / "g.json"
    graph.write_text("{}")
    result = runner.invoke(app, ["pipelines", "run", str(graph), "--secret", "NOPE"], env=TEST_ENV)
    assert result.exit_code == 1 and "NOPE" in result.output and "post" not in api


def test_status_shows_counts_cards_cost_and_outputs(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "status", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert api["get"] == "/pipelines/runs/run1"
    for text in ("scout", "Waiting on you", "Keep or drop each idea.", "$0.25", "report/report.md"):
        assert text in result.output


def test_status_as_json(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "status", "run1", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0 and json.loads(result.output)["id"] == "run1"


def test_stop(api: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "stop", "run1"], env=TEST_ENV)
    assert result.exit_code == 0 and api["post"] == ("/pipelines/runs/run1/stop", None)


def test_status_shows_bracketed_text_as_written(api: Dict[str, Any], monkeypatch) -> None:
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
    result = runner.invoke(app, ["pipelines", "status", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "bad [type=missing]" in result.output and "Choose [keep/drop]" in result.output
    assert "report/[/red].md" in result.output


@pytest.mark.parametrize("name", ["../evil", "D:evil"])
def test_outputs_refuse_a_name_that_leaves_the_directory(monkeypatch, tmp_path, name) -> None:
    run = {**RUN, "outputs": [{"name": name, "node": "report", "bytes": 1}]}
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


def test_cards_lists_what_waits_and_how_to_answer_each(review: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "cards", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert review["get"] == "/pipelines/runs/run1"
    for text in (
        "review (job review.t1), review: Keep or drop each idea.",
        "- b",
        "gate (job graph), choice",
        "Options: yes, no",
        "ask (job graph), question: Which repo?",
        "prime pipelines reply run1 --node gate --job graph --choice OPTION",
        "--note TEXT",
    ):
        assert text in result.output
    assert review["posts"] == []


def test_cards_say_why_a_reply_was_not_applied(review: Dict[str, Any]) -> None:
    review["run"] = {
        **RUN,
        "state": {"cards": [], "skipped_controls": [{"seq": 3, "error": "KeyError: [job]"}]},
    }
    result = runner.invoke(app, ["pipelines", "cards", "run1"], env=TEST_ENV)
    assert result.exit_code == 0, result.output
    assert "control 3" in result.output and "KeyError: [job]" in result.output
    assert "Nothing is waiting on you." in result.output


def test_cards_as_json(review: Dict[str, Any]) -> None:
    result = runner.invoke(app, ["pipelines", "cards", "run1", "-o", "json"], env=TEST_ENV)
    assert result.exit_code == 0 and json.loads(result.output)["cards"] == CARDS


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
        ([], "3 cards match"),
        (["--job", "graph", "--choice", "yes"], "2 cards match"),
        (["--node", "nope"], "no waiting card matches"),
        (["--node", "gate", "--choice", "maybe"], "choose one of: yes, no"),
        (["--node", "gate", "--keep", "a"], "it has no items"),
        (["--node", "review", "--choice", "yes"], "no options"),
        (["--node", "review"], "one of --keep, --drop or --keep-all"),
        (["--node", "review", "--keep", "a", "--drop", "b"], "one of --keep"),
        (["--node", "review", "--keep", "z"], "not an item of this card: z"),
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


def test_refresh_and_drain(review: Dict[str, Any]) -> None:
    assert (
        runner.invoke(app, ["pipelines", "refresh", "run1", "traces"], env=TEST_ENV).exit_code == 0
    )
    assert runner.invoke(app, ["pipelines", "drain", "run1"], env=TEST_ENV).exit_code == 0
    assert review["posts"] == [
        ("/pipelines/runs/run1/control", {"control": {"type": "refresh", "node": "traces"}}),
        ("/pipelines/runs/run1/control", {"control": {"type": "drain"}}),
    ]
    assert "get" not in review


def test_control_paths_encode_the_run_id(review: Dict[str, Any]) -> None:
    assert runner.invoke(app, ["pipelines", "drain", "r/1"], env=TEST_ENV).exit_code == 0
    assert review["posts"][0][0] == "/pipelines/runs/r%2F1/control"
