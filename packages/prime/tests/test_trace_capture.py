"""`prime inference trace-capture` drives the platform API, so the team travels
as a `teamId` param and a 403 has to read as a permission problem rather than a
stack trace. The effective state is the point of the listing: `blocked` must
never render as `off`."""

import json
from typing import Any, Dict, List, Optional

import pytest
from prime_cli.api.trace_capture import (
    BASE_LORA_ID,
    TraceCaptureClient,
    TraceCaptureForbiddenError,
    TraceCaptureTarget,
    synthetic_run_id,
)
from prime_cli.client import APIError
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV: Dict[str, str] = {
    "COLUMNS": "200",
    "LINES": "50",
    "NO_COLOR": "1",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}

ALL_STATES = (
    "on",
    "off",
    "blocked_privacy_opt_out",
    "blocked_storage_full",
    "ineligible",
)


class _TeamConfig:
    team_id = "team-1"


class _PersonalConfig:
    team_id = None


def _target(
    *,
    model: str = "z-ai/glm-5.3",
    state: str = "on",
    lora_id: str = BASE_LORA_ID,
    sample_rate: Optional[float] = 0.1,
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "loraId": lora_id,
        "state": state,
        "enabled": state not in ("off", "ineligible"),
        "eligible": state != "ineligible",
        "sampleRate": sample_rate,
        "updatedAt": "2026-10-20T00:00:00Z",
    }
    payload.update(extra)
    return payload


class DummyCaptureClient:
    """Stands in for the API client; records what the command asked for."""

    instances: List["DummyCaptureClient"] = []

    def __init__(self, api_client: Any) -> None:
        self.api_client = api_client
        self.calls: Dict[str, Any] = {}
        DummyCaptureClient.instances.append(self)

    targets: List[Dict[str, Any]] = []
    set_result: Optional[Dict[str, Any]] = None
    raises: Optional[Exception] = None

    def list_targets(self, params: Optional[Dict[str, Any]] = None) -> List[TraceCaptureTarget]:
        self.calls["list_targets"] = params
        if self.raises is not None:
            raise self.raises
        return [TraceCaptureTarget.model_validate(t) for t in self.targets]

    def set_target(self, **kwargs: Any) -> Optional[TraceCaptureTarget]:
        self.calls["set_target"] = kwargs
        if self.raises is not None:
            raise self.raises
        if self.set_result is None:
            return None
        return TraceCaptureTarget.model_validate(self.set_result)


@pytest.fixture()
def capture_client(monkeypatch: pytest.MonkeyPatch) -> type[DummyCaptureClient]:
    DummyCaptureClient.instances = []
    DummyCaptureClient.targets = []
    DummyCaptureClient.set_result = None
    DummyCaptureClient.raises = None
    monkeypatch.setattr("prime_cli.commands.trace_capture.APIClient", lambda: object())
    monkeypatch.setattr("prime_cli.commands.trace_capture.Config", _TeamConfig)
    monkeypatch.setattr("prime_cli.commands.trace_capture.TraceCaptureClient", DummyCaptureClient)
    return DummyCaptureClient


def _last(client: type[DummyCaptureClient]) -> DummyCaptureClient:
    assert client.instances, "the command never built a client"
    return client.instances[-1]


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_passes_the_team_as_a_param(capture_client) -> None:
    capture_client.targets = [_target()]

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert _last(capture_client).calls["list_targets"] == {"teamId": "team-1"}


def test_list_renders_every_state_distinctly(capture_client) -> None:
    capture_client.targets = [
        _target(model=f"vendor/model-{i}", state=state) for i, state in enumerate(ALL_STATES)
    ]

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    # A word per state, so the table survives --plain / NO_COLOR.
    assert "blocked: storage full" in output
    assert "blocked: privacy opt-out" in output
    assert "ineligible" in output
    # ...and each has an explanation, so "blocked" never reads as "off".
    assert "the team's trace storage limit" in output
    assert "trace privacy" in output
    assert "not approved for capture" in output


def test_list_never_collapses_an_unknown_state(capture_client) -> None:
    capture_client.targets = [_target(state="blocked_future_reason")]

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    # Rendered verbatim rather than mapped onto one of ours.
    assert "blocked_future_reason" in output
    assert "older than the platform" in output


def test_list_hints_the_matching_traces_command(capture_client) -> None:
    capture_client.targets = [
        _target(model="z-ai/glm-5.3", state="on"),
        _target(model="qwen/qwen3-8b", state="on", lora_id="adapter-7"),
        _target(model="internal/fast", state="off"),
    ]

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "prime traces list --run-id inference/z-ai/glm-5.3" in output
    assert "prime traces list --runs --run-type inference" in output


def test_list_hints_how_to_turn_an_off_target_on(capture_client) -> None:
    capture_client.targets = [_target(model="qwen/qwen3-8b", state="off", lora_id="adapter-7")]

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "prime inference trace-capture enable qwen/qwen3-8b --lora-id adapter-7" in output


def test_list_omits_the_read_hint_when_nothing_is_capturing(capture_client) -> None:
    capture_client.targets = [_target(state="blocked_storage_full")]

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "prime traces list --run-id" not in output


def test_list_json_carries_the_synthetic_run_id(capture_client) -> None:
    capture_client.targets = [
        _target(model="qwen/qwen3-8b", lora_id="adapter-7", state="on"),
        _target(model="z-ai/glm-5.3", state="blocked_storage_full"),
    ]

    result = runner.invoke(app, ["inference", "trace-capture", "list", "-o", "json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["total"] == 2
    assert payload["targets"][0]["run_id"] == "inference/qwen/qwen3-8b:adapter-7"
    assert payload["targets"][1]["state"] == "blocked_storage_full"


def test_list_escapes_server_supplied_model_ids(capture_client) -> None:
    capture_client.targets = [_target(model="[red]vendor/evil[/red]")]

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "[red]vendor/evil[/red]" in output


def test_list_reports_an_empty_team_without_a_table(capture_client) -> None:
    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "No capture targets for this team" in output


def test_list_forbidden_explains_the_permission(capture_client) -> None:
    capture_client.raises = TraceCaptureForbiddenError("Not a member of this team")

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)

    assert result.exit_code == 1
    stderr = strip_ansi(result.stderr)
    assert "Not allowed: Not a member of this team" in stderr
    assert "membership of the team" in stderr
    assert "Traceback" not in result.output


def test_list_json_keeps_errors_off_stdout(capture_client) -> None:
    capture_client.raises = APIError("HTTP 500: upstream exploded")

    result = runner.invoke(app, ["inference", "trace-capture", "list", "-o", "json"], env=TEST_ENV)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "upstream exploded" in strip_ansi(result.stderr)


def test_list_requires_a_team(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("prime_cli.commands.trace_capture.Config", _PersonalConfig)

    def fail() -> Any:
        pytest.fail("a personal context must not reach the API")

    monkeypatch.setattr("prime_cli.commands.trace_capture.APIClient", fail)

    result = runner.invoke(app, ["inference", "trace-capture", "list"], env=TEST_ENV)

    assert result.exit_code == 1
    stderr = strip_ansi(result.stderr)
    assert "no team is selected" in stderr
    assert "prime switch" in stderr


# ---------------------------------------------------------------------------
# enable / disable
# ---------------------------------------------------------------------------


def test_enable_sends_camelcase_body_with_the_team(capture_client) -> None:
    capture_client.set_result = _target(state="on", sample_rate=0.25)

    result = runner.invoke(
        app,
        [
            "inference",
            "trace-capture",
            "enable",
            "z-ai/glm-5.3",
            "--sample-rate",
            "0.25",
            "--yes",
        ],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert _last(capture_client).calls["set_target"] == {
        "team_id": "team-1",
        "model": "z-ai/glm-5.3",
        "lora_id": BASE_LORA_ID,
        "enabled": True,
        "sample_rate": 0.25,
    }
    assert "Trace capture enabled for z-ai/glm-5.3" in output
    assert "Effective state: on" in output
    assert "prime traces list --run-id inference/z-ai/glm-5.3" in output


def test_enable_reports_a_blocked_outcome_rather_than_plain_success(capture_client) -> None:
    capture_client.set_result = _target(state="blocked_storage_full")

    result = runner.invoke(
        app,
        ["inference", "trace-capture", "enable", "z-ai/glm-5.3", "--yes"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "Effective state: blocked: storage full" in output
    assert "nothing is being stored" in output
    # The read hint would be a lie while nothing is stored.
    assert "prime traces list --run-id" not in output


def test_enable_targets_an_adapter_with_its_own_run_id(capture_client) -> None:
    capture_client.set_result = _target(model="qwen/qwen3-8b", lora_id="adapter-7", state="on")

    result = runner.invoke(
        app,
        [
            "inference",
            "trace-capture",
            "enable",
            "qwen/qwen3-8b",
            "--lora-id",
            "adapter-7",
            "--yes",
        ],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert _last(capture_client).calls["set_target"]["lora_id"] == "adapter-7"
    assert "adapter adapter-7" in output
    assert "prime traces list --run-id inference/qwen/qwen3-8b:adapter-7" in output


def test_enable_confirms_retention_before_writing(capture_client) -> None:
    result = runner.invoke(
        app,
        ["inference", "trace-capture", "enable", "z-ai/glm-5.3"],
        input="n\n",
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "retains the prompts and completions" in output
    assert "Cancelled." in output
    assert not capture_client.instances


def test_enable_json_requires_yes_and_keeps_stdout_clean(capture_client) -> None:
    result = runner.invoke(
        app,
        ["inference", "trace-capture", "enable", "z-ai/glm-5.3", "-o", "json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "--output json requires --yes" in strip_ansi(result.stderr)
    assert not capture_client.instances


def test_enable_json_emits_only_the_target(capture_client) -> None:
    capture_client.set_result = _target(state="blocked_privacy_opt_out")

    result = runner.invoke(
        app,
        ["inference", "trace-capture", "enable", "z-ai/glm-5.3", "--yes", "-o", "json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["state"] == "blocked_privacy_opt_out"
    assert payload["run_id"] == "inference/z-ai/glm-5.3"


def test_enable_without_an_echoed_target_does_not_invent_a_state(capture_client) -> None:
    capture_client.set_result = None

    result = runner.invoke(
        app,
        ["inference", "trace-capture", "enable", "z-ai/glm-5.3", "--yes"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "Effective state" not in output
    assert "trace-capture list" in output


def test_disable_sends_enabled_false_without_a_sample_rate(capture_client) -> None:
    capture_client.set_result = _target(state="off", sample_rate=None)

    result = runner.invoke(
        app,
        ["inference", "trace-capture", "disable", "z-ai/glm-5.3"],
        env=TEST_ENV,
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert _last(capture_client).calls["set_target"] == {
        "team_id": "team-1",
        "model": "z-ai/glm-5.3",
        "lora_id": BASE_LORA_ID,
        "enabled": False,
        "sample_rate": None,
    }
    assert "Trace capture disabled for z-ai/glm-5.3" in output
    assert "Effective state: off" in output


def test_disable_forbidden_names_team_admin(capture_client) -> None:
    capture_client.raises = TraceCaptureForbiddenError("Team admin required")

    result = runner.invoke(
        app, ["inference", "trace-capture", "disable", "z-ai/glm-5.3"], env=TEST_ENV
    )

    assert result.exit_code == 1
    stderr = strip_ansi(result.stderr)
    assert "team ADMIN" in stderr
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# API client wire shape
# ---------------------------------------------------------------------------


class FakeAPIClient:
    """Minimal stand-in for APIClient: records the request, returns a payload."""

    def __init__(self, response: Any = None, error: Optional[Exception] = None) -> None:
        self.response = response if response is not None else {}
        self.error = error
        self.requests: List[Dict[str, Any]] = []

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self.requests.append({"method": "GET", "endpoint": endpoint, "params": params})
        if self.error is not None:
            raise self.error
        return self.response

    def request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        self.requests.append({"method": method, "endpoint": endpoint, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response


def test_client_reads_without_the_api_v1_prefix() -> None:
    api = FakeAPIClient({"targets": [_target()]})

    targets = TraceCaptureClient(api).list_targets({"teamId": "team-1"})

    # APIClient adds /api/v1 itself.
    assert api.requests == [
        {"method": "GET", "endpoint": "/inference/trace-capture", "params": {"teamId": "team-1"}}
    ]
    assert targets[0].model == "z-ai/glm-5.3"
    assert targets[0].lora_id == BASE_LORA_ID
    assert targets[0].sample_rate == 0.1


def test_client_accepts_either_listing_key() -> None:
    assert TraceCaptureClient(FakeAPIClient({"items": [_target()]})).list_targets() != []
    assert TraceCaptureClient(FakeAPIClient({"targets": []})).list_targets() == []
    assert TraceCaptureClient(FakeAPIClient({})).list_targets() == []


def test_client_writes_camelcase_body_over_put() -> None:
    api = FakeAPIClient({"target": _target(state="on")})

    target = TraceCaptureClient(api).set_target(
        team_id="team-1",
        model="z-ai/glm-5.3",
        lora_id="adapter-7",
        enabled=True,
        sample_rate=0.5,
    )

    assert api.requests == [
        {
            "method": "PUT",
            "endpoint": "/inference/trace-capture",
            "json": {
                "teamId": "team-1",
                "model": "z-ai/glm-5.3",
                "loraId": "adapter-7",
                "enabled": True,
                "sampleRate": 0.5,
            },
        }
    ]
    assert target is not None and target.state == "on"


def test_client_omits_an_unset_sample_rate() -> None:
    api = FakeAPIClient({})

    assert TraceCaptureClient(api).set_target(team_id="t", model="m", enabled=False) is None
    assert "sampleRate" not in api.requests[0]["json"]
    assert api.requests[0]["json"]["loraId"] == BASE_LORA_ID


def test_client_turns_403_into_a_permission_error() -> None:
    api = FakeAPIClient(error=APIError("HTTP 403: Only team admins can change trace capture"))

    with pytest.raises(TraceCaptureForbiddenError, match="Only team admins"):
        TraceCaptureClient(api).set_target(team_id="t", model="m", enabled=True)


def test_client_names_the_action_on_other_failures() -> None:
    api = FakeAPIClient(error=APIError("HTTP 500: boom"))

    with pytest.raises(APIError, match="Failed to list trace-capture targets"):
        TraceCaptureClient(api).list_targets()


@pytest.mark.parametrize(
    ("model", "lora_id", "expected"),
    [
        ("z-ai/glm-5.3", BASE_LORA_ID, "inference/z-ai/glm-5.3"),
        ("qwen/qwen3-8b", "adapter-7", "inference/qwen/qwen3-8b:adapter-7"),
        ("frontier/dep-x/slug", "", "inference/frontier/dep-x/slug"),
    ],
)
def test_synthetic_run_id_matches_the_gateway(model, lora_id, expected) -> None:
    assert synthetic_run_id(model, lora_id) == expected
