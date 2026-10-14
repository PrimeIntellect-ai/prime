# prime-dev-sandboxes

Typed Python SDK for the Prime Intellect dev-sandboxes API
(`/api/v1/dev-sandboxes`).

Dev sandboxes are long-lived, per-task development environments booted from
checked-in templates (for example `platform-dev`) at a specific `ref` (PR URL,
branch, or SHA). Long actions (create, exec, delete) are admitted as
operations; poll `GET /api/v1/dev-sandboxes/operations/{id}` until terminal.

This package speaks the API only. It contains no local lifecycle engine —
the platform backend drives allocation, boot, readiness probes, tunnels, and
teardown.

```python
from prime_dev_sandboxes import DevSandboxClient

client = DevSandboxClient()  # uses PRIME_API_KEY / ~/.prime/config.json
created = client.create(template="platform-dev", ref="origin/develop", ttl_minutes=120)
operation = client.wait_for_operation(created.operation_id)
env = client.get(created.environment_id)
print(env.state, env.urls, env.expiry)
```

Authentication follows the other `prime-*` SDK packages: `PRIME_API_KEY`
(or `~/.prime/config.json`), with `PRIME_API_BASE_URL` / `PRIME_BASE_URL`
and `PRIME_CONTEXT` overlays resolved the same way.

Error taxonomy: every failure raised as `DevSandboxError` carries a code from
five groups (`admission`, `template`, `infrastructure`, `readiness`,
`lifecycle`), a stage, a retryable flag, provider code, operation id, and
owned resource ids — mirroring the backend's structured error responses.
