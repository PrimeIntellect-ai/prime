"""Shared response samples for the sync and async client contract tests."""

# Keep this representative of the wire-level summary contract.
SUMMARY = {
    "trace_id": "8d3f1a2b",
    "upload_id": "5ee85e41",
    "episode_id": None,
    "created_at": "2026-07-20T18:02:11.482Z",
    "ingested_at": "2026-07-20T18:06:02.117Z",
    "run_id": "run_9f3k2m",
    "environment_id": "terminal-bench-2",
    "model": {"provider": "prime", "id": "deepseek-v4-flash"},
    "task_id": "tb2-0187",
    "agent_name": "solver",
    "score": {"reward": 0.85, "outcome": "done"},
    "execution": {"has_error": False, "is_truncated": False},
    "duration_ms": 215537,
    "total_tokens": 84213,
    "size_bytes": 417284,
    "context": {"source": "hosted_eval"},
}

# Keep this representative of the wire-level episode contract.
EPISODE = {
    "episode_id": "ep-1",
    "upload_id": "5ee85e41",
    "schema_version": 1,
    "created_at": "2026-07-20T18:02:11.482Z",
    "ingested_at": "2026-07-20T18:06:02.117Z",
    "run_id": "run_9f3k2m",
    "environment_id": "terminal-bench-2",
    "outcome": "done",
    "has_error": False,
    "error": {"type": None, "message": None},
}

EMPTY_AGGREGATE = {
    "trace_count": 0,
    "total_tokens": 0,
    "total_duration_ms": 0,
    "any_trace_error": False,
    "agent_names": [],
}

RESERVED_TRACE_ID = "trace?with#reserved%chars and space"
ENCODED_TRACE_PATH = b"/api/v1/traces/trace%3Fwith%23reserved%25chars%20and%20space"
RESERVED_EPISODE_ID = "episode?with#reserved%chars and space"
ENCODED_EPISODE_PATH = b"/api/v1/episodes/episode%3Fwith%23reserved%25chars%20and%20space"

UNAVAILABLE = {"error": {"code": "storage_unavailable", "message": "try again"}}
