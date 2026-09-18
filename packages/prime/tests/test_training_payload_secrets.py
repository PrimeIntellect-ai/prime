from prime_cli.api.training import build_payload_from_toml


def test_secrets_map_rides_on_the_payload() -> None:
    payload = build_payload_from_toml(
        {"trainer": {}},
        secrets={"OPENAI_API_KEY": "sk-abc", "JUDGE_API_KEY": "j-xyz"},
    )

    assert payload["secrets"] == {"OPENAI_API_KEY": "sk-abc", "JUDGE_API_KEY": "j-xyz"}


def test_secrets_key_omitted_when_empty() -> None:
    # The backend treats absent and empty the same way, so don't send a
    # key at all rather than an empty object.
    assert "secrets" not in build_payload_from_toml({"trainer": {}}, secrets={})
    assert "secrets" not in build_payload_from_toml({"trainer": {}}, secrets=None)
    assert "secrets" not in build_payload_from_toml({"trainer": {}})


def test_reserved_credentials_keep_their_dedicated_fields() -> None:
    # WANDB_API_KEY / HF_TOKEN travel as wandbApiKey / hfToken. The
    # backend 422s on either name inside `secrets`, so the two must not
    # arrive by both routes.
    payload = build_payload_from_toml(
        {"trainer": {}},
        wandb_api_key="w-1",
        hf_token="h-1",
        secrets={"OPENAI_API_KEY": "sk-abc"},
    )

    assert payload["wandbApiKey"] == "w-1"
    assert payload["hfToken"] == "h-1"
    assert payload["secrets"] == {"OPENAI_API_KEY": "sk-abc"}


def test_secrets_stay_out_of_the_config_blob() -> None:
    # `config` is shipped verbatim to the pods as TOML; credentials must
    # only reach them via the per-run k8s Secret.
    cfg = {"trainer": {"model": {"name": "Qwen/Qwen3-4B"}}}
    payload = build_payload_from_toml(cfg, secrets={"OPENAI_API_KEY": "sk-abc"})

    assert "sk-abc" not in str(payload["config"])
