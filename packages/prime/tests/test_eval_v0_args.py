"""The frozen v0 eval argv (hosted sandboxes still emit it) pops cleanly out of argv."""

from prime_cli.verifiers_bridge import _pop_v0_eval_args


def test_platform_sandbox_argv_converts():
    # the exact shape platform's script_generator emits for env_id evals
    args = [
        "--num-examples",
        "10",
        "--rollouts-per-example",
        "4",
        "--save-results",
        "--debug",
        "--model",
        "org/model-a",
        "--verbose",
        "--env-args",
        '{"difficulty": "hard"}',
        "--sampling-args",
        '{"max_tokens": 32}',
        "--max-concurrent",
        "8",
        "--max-retries",
        "3",
        "--state-columns",
        "a,b",
        "--independent-scoring",
        "--header",
        "X-PI-Job-Id: j1",
        "--api-client-type",
        "openai_chat_completions",
        "--api-base-url",
        "https://api.pinference.ai/api/v1",
        "--api-key-var",
        "PRIME_API_KEY",
    ]
    fields, tui_disabled = _pop_v0_eval_args(args)

    assert tui_disabled
    # v1-native flags stay on the command line
    assert args == ["--model", "org/model-a", "--verbose", "--max-concurrent", "8"]
    assert fields == {
        "num_examples": 10,
        "rollouts_per_example": 4,
        "independent_scoring": True,
        "env_args": {"difficulty": "hard"},
        "sampling_args": {"max_tokens": 32},
        "max_retries": "3",
        "state_columns": "a,b",
        "api_client_type": "openai_chat_completions",
        "api_base_url": "https://api.pinference.ai/api/v1",
        "api_key_var": "PRIME_API_KEY",
        "header": ["X-PI-Job-Id: j1"],
    }


def test_pure_v1_argv_untouched():
    args = ["--model", "m", "-n", "5", "--sampling.max-tokens", "9", "--no-rich"]
    fields, tui_disabled = _pop_v0_eval_args(args)
    assert (fields, tui_disabled) == ({}, False)
    assert args == ["--model", "m", "-n", "5", "--sampling.max-tokens", "9", "--no-rich"]


def test_config_run_tui_flag_strips_without_conversion():
    # platform config runs append only --skip-upload (typer) and ${TUI_FLAG}
    args = ["--disable-tui"]
    fields, tui_disabled = _pop_v0_eval_args(args)
    assert (fields, tui_disabled) == ({}, True)
    assert args == []
