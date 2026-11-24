# Prime Sandboxes SDK

Lightweight Python SDK for managing Prime Intellect sandboxes - secure remote code execution environments.

## Features

- **Synchronous and async clients** - Use with sync or async/await code
- **Full sandbox lifecycle** - Create, list, execute commands, upload/download files, delete
- **Type-safe** - Full type hints and Pydantic models
- **Authentication caching** - Automatic token management
- **Bulk operations** - Create and manage multiple sandboxes efficiently
- **No CLI dependencies** - Pure SDK, ~50KB installed

## Installation

```bash
uv pip install prime-sandboxes
```

Or with pip:

```bash
pip install prime-sandboxes
```

## Quick Start

```python
from prime_sandboxes import APIClient, SandboxClient, CreateSandboxRequest, StartCommand

# Initialize
client = APIClient(api_key="your-api-key")
sandbox_client = SandboxClient(client)

# Create a VM-backed sandbox.
request = CreateSandboxRequest(
    name="my-sandbox",
    docker_image="python:3.11-slim",
    cpu_cores=2,
    memory_gb=4,
)

sandbox = sandbox_client.create(request)
print(f"Created: {sandbox.id}")

# Boot commands use a structured argv contract; no shell is implied.
vm = sandbox_client.create(CreateSandboxRequest(
    name="vm-workload",
    docker_image="user-1/vm-image:latest",
    start_command=StartCommand(
        executable="/worker",
        args=["--platform", "linux/amd64"],
    ),
))


sandbox_client.wait_for_creation(sandbox.id)

# Wait for checkpoint durability before restoring.
checkpoint = sandbox_client.checkpoint(sandbox.id)
checkpoint = sandbox_client.wait_for_checkpoint(checkpoint.id, timeout_seconds=300)
restored = sandbox_client.create(CreateSandboxRequest(
    name="restored-sandbox", checkpoint_id=checkpoint.id
))

# Execute commands
result = sandbox_client.execute_command(sandbox.id, "python --version")
print(result.stdout)

# Clean up
sandbox_client.delete(sandbox.id)
```

## Async Usage

```python
import asyncio
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

async def main():
    async with AsyncSandboxClient(api_key="your-api-key") as client:
        # Create sandbox
        sandbox = await client.create(CreateSandboxRequest(
            name="async-sandbox",
            docker_image="python:3.11-slim",
        ))

        # Wait and execute
        await client.wait_for_creation(sandbox.id)
        result = await client.execute_command(sandbox.id, "echo 'Hello from async!'")
        print(result.stdout)

        # Wait for checkpoint durability
        checkpoint = await client.checkpoint(sandbox.id)
        durable = await client.wait_for_checkpoint(checkpoint.id, timeout_seconds=300)
        print(durable.state)

        # Clean up
        await client.delete(sandbox.id)

asyncio.run(main())
```

Concurrent waits on the same client automatically share cross-sandbox status
requests, with up to 100 checkpoint IDs per request:

```python
checkpoints = await asyncio.gather(*(client.checkpoint(s.id) for s in sandboxes))
durable = await asyncio.gather(*(client.wait_for_checkpoint(c.id) for c in checkpoints))
```

Sync waits from concurrent threads share requests too. `get_checkpoints(ids)`
provides an explicit batch lookup with `checkpoints` and per-ID `errors`. Waits
fall back to individual lookups on platforms without the batch endpoint.

## List Platform Images

Use a platform admin or manager key with sandbox-read access to list platform images:

```python
from prime_sandboxes import ImageBuildStatus, ImageClient

page = ImageClient().list(platform=True)
completed_images = [
    image.display_ref
    for image in page.data
    if image.status == ImageBuildStatus.COMPLETED
]
```

## Image Builds

Dockerfile builds create VM artifacts on `linux/amd64`. The
initial response includes `upload_url` and `expires_in`; upload the build context
before calling `start_build`.

Source-image requests build VM artifacts directly from allowed public registry
images. They do not return upload metadata. A single source returns `build_id`
and `build_ids`. Comma-separated sources return `BulkBuildImageResponse` with
ordered `results`: each entry has `source_image`, `build` (a `BuildImageResponse`
or `None`), `error`, and `retryable`. There are no `success` or `failed` fields.

The server uses mixed wire casing: `build_id`, `upload_url`, and `expires_in`,
but `buildIds`, `fullImagePath`, and `sourceImage`. SDK attributes use snake_case.
The `transfer_image` method remains a compatibility name for `POST /images/build`:

```python
from prime_sandboxes import ImageClient

images = ImageClient()
response = images.transfer_image("ubuntu:22.04")
print(response.build_ids)
```

All image builds support only `linux/amd64`. Docker Hub sources become public,
org-less platform images automatically. Docker Hub source builds do not accept a
custom destination, team, or private visibility. One comma-separated request
cannot mix Docker Hub with other registries. Explicit non-Docker-Hub public
registries can still use personal or team ownership, custom destinations, and
public or private visibility. Allowed registries are Docker Hub, `ghcr.io`,
`quay.io`, `public.ecr.aws`, `registry.k8s.io`, and `mcr.microsoft.com`.
Google-hosted registries are rejected. Docker-Hub-only multi-source requests
preserve source names and tags and force PUBLIC platform scope.

Use `prime images push --source-image <reference>` for one or comma-separated
sources, or `prime images push-bulk` for manifests. Dockerfile platform
publishing uses `prime images push <name>:<tag> --platform-image`; the primary
build creates its VM artifact without a second publishing step.

## Volumes

Volumes are persistent ext4 block volumes mounted into VM sandboxes. A volume
starts empty; every sandbox that mounts it attaches the current head, and its
writes are private to that sandbox until a publish becomes DURABLE. Names are
resolved in the caller's scope: pass `team_id` (or set a team in the client
config) for team scope, otherwise the personal scope applies. A
`<namespace>/<name>` ref names a PUBLIC volume owned by another account and
is sent with its slash literal; the namespace may be a user id or slug, a
team slug, or `team-<teamId>`. `list` stays own-scope and responses include
each volume's `namespace` and `visibility`, plus `total`, `page`, `per_page`,
and `has_next`. A response's `namespace` is the owner's user or team slug, or
the id form (the user id, or `team-<teamId>`) when it has no slug, its slug
starts with `team-`, or its slug is an existing user id (that namespace
resolves to that user, not to the owner). `get` by a foreign ref returns only
`name`, `namespace`, `visibility`, `size_gb`, `occupied_bytes`, and
`created_at`.

```python
from prime_sandboxes import APIClient, SandboxClient, VolumeVisibility

volumes = SandboxClient(APIClient()).volumes
volume = volumes.create("dataset", team_id="team-id")
volume = volumes.create("dataset", size_gb=512)  # 100 GiB when omitted
page = volumes.list(team_id="team-id", page=1, per_page=50)
volume = volumes.get("dataset", team_id="team-id")
volume = volumes.grow("dataset", 512)
volume = volumes.set_visibility("dataset", VolumeVisibility.PUBLIC)
volumes.delete("dataset", team_id="team-id")  # 204, returns None
```

Omit `team_id` (and the team in the client config) to work in the personal
scope instead. `volumes` is a namespace on the sandbox client: it reaches
the platform through the client's own transport and owns no transport or
lifecycle of its own. `size_gb` is the volume's capacity in whole GiB;
`Volume.size_gb` is null when the volume uses the default base-image size.
`grow` is grow-only: the server rejects a size at or below the current
effective size, and the new size applies to sandboxes created after the
change — running attachments keep their size.

`set_visibility` makes a volume PUBLIC or PRIVATE for its owner: its user for
a personal volume, any team member for a team volume. A PUBLIC volume mounts
and syncs from any account through its `<namespace>/<name>` ref; volumes stay
unlisted, so a recipient needs the ref. Recipients read the shared head and
their writes stay in their own sandbox — they cannot publish, and their sync
always discards local writes, whatever `discard` says. Making a volume PRIVATE
stops new recipient mounts, syncs, and gets; running recipient sandboxes keep
their attachment.

Mount volumes when creating a sandbox. Each mount names the volume and the
absolute guest path to mount it at; a read-only mount cannot publish:

```python
from prime_sandboxes import APIClient, CreateSandboxRequest, SandboxClient, VolumeMount

request = CreateSandboxRequest(
    name="trainer",
    docker_image="ubuntu:22.04",
    volumes=[VolumeMount(name="dataset", mount_path="/data", read_only=True)],
)
sandbox = SandboxClient(APIClient()).create(request)
```

A sandbox sees the volume head from the moment it was created (the attached
head) and never sees another sandbox's unpublished writes. To see a newer head,
recreate the sandbox. `publish` makes a sandbox's own volume contents durable
and moves the volume head:

```python
from prime_sandboxes import APIClient, SandboxClient

volumes = SandboxClient(APIClient()).volumes
checkpoint = volumes.publish("sandbox-id", "dataset")  # waits for DURABLE
checkpoint = volumes.publish("sandbox-id", "dataset", wait=False)
```

By default `publish` waits for the new head checkpoint to become DURABLE and
returns it; pass `wait=False` to get the pending checkpoint immediately and
poll it yourself. On servers that report a checkpoint's volume-merge status
the same wait also covers the merge: `RuntimeError` carries `merge_error`
when the merge failed, and a publish or merge still unfinished after the
wait budget (300 seconds by default) raises `TimeoutError`. The server
rejects the publish with HTTP 404 when the sandbox has no writable attachment
with that name, HTTP 400 when the mount is read-only, and HTTP 409 when
another sandbox's publish for the same volume is already in flight. A head
that moved since the sandbox was created is also HTTP 409.
`AsyncSandboxClient` exposes the same operations with `await` on its own
`volumes` namespace; the parent client's `aclose()` closes its owned
transport and polling work, while the namespace owns neither.

Deleting a volume that is attached to a sandbox in any active state (PENDING,
PROVISIONING, RUNNING, PAUSED, or TERMINATING) fails with HTTP 409. Any team
member can delete a team volume. Volume deletes retry transient 5xx
automatically.

## Authentication

The SDK looks for credentials in this order:

1. **Direct parameter**: `APIClient(api_key="sk-...")`
2. **Environment variable**: `export PRIME_API_KEY="sk-..."`
3. **Config file**: `~/.prime/config.json` (created by `prime login` CLI command)

## Advanced Features

### Environment Variables and Secrets

```python
# Create sandbox with environment variables and secrets
request = CreateSandboxRequest(
    name="my-sandbox",
    docker_image="python:3.11-slim",
    environment_vars={
        "DEBUG": "true",
        "LOG_LEVEL": "info"
    },
    secrets={
        "API_KEY": "sk-secret-key-here",
        "DATABASE_PASSWORD": "super-secret-password"
    }
)

sandbox = sandbox_client.create(request)
```

**Note:** Secrets are never displayed in logs or outputs. When retrieving sandbox details, only the secret keys are shown with values masked as `***`.

### Run Commands as a Guest User

Commands use the sandbox's configured user (normally root) when `user` is omitted.
Select an existing account in the guest image per command:

```python
result = sandbox_client.execute_command(sandbox.id, "id", user="ubuntu")
# Async execute_command, open_process, and sync/async start_background_job also accept user.
```

Empty or unknown usernames fail; accounts are not created automatically. Commands
use the account's UID, GID, and supplementary groups. `HOME`, `USER`, and `LOGNAME`
follow that account unless explicitly overridden through `env`. Working-directory
selection stays unchanged; pass `working_dir` if the image default is inaccessible.

Requires a sandboxd version supporting command-level users. Roll out sandboxd to
all reachable guests before releasing this SDK: older servers can ignore the
protobuf field and execute as their default user. Existing guests must be upgraded
or recreated before using this option.

### Native Process Output Recovery

`AsyncSandboxClient.open_process()` resumes interrupted output streams by session
UUID and output sequence. Replayed chunks are deduplicated across stdout and
stderr, and the final sequence is checked before accepting the exit code.

Native sessions lease the bounded command connection pool for their lifetime,
including reconnects. Connections are reused across commands, with capacity
accounted for stdin and signal requests too, so blocked input cannot starve
signals. Keep the sandbox client open until its process handles have been closed.

sandboxd retains up to 4 MiB per command and 64 MiB across the guest, including
record overhead. Connected readers apply backpressure rather than losing unread
output. Disconnected output can be evicted; recovery then raises
`ProcessOutputLostError` instead of returning incomplete output as success.
Forced pipe closure also raises this error when complete output cannot be proven.
Replay uses the existing guest-local session lifetime and does not survive a
daemon restart or session expiration.

Roll out sandboxd before releasing this SDK. Uninterrupted streams work with older
daemons; an interruption fails explicitly when the server cannot replay output.
Regenerate the Python protocol bindings from the platform checkout with
`bash packages/prime-sandboxes/tools/generate-command-session.sh /path/to/platform`.

### Fork

`fork()` creates a new sandbox from the source's filesystem. By default it
checkpoints the source's disk, waits for the checkpoint to be DURABLE, and
creates the fork with the source's shape, team, region, timeout, and labels; the
source keeps running:

```python
forked = sandbox_client.fork(sandbox.id, name="forked", cpu_cores=4)
sandbox_client.wait_for_creation(forked.id)
```

`with_memory=True` also forks the source's memory and running processes: the
source is paused, the fork is created from that pause, and the source is resumed.
The fork takes the paused sandbox's shape, environment, and secrets, so
`cpu_cores` and `memory_gb` cannot be combined with it:

```python
forked = sandbox_client.fork(sandbox.id, name="forked", with_memory=True)
```

`AsyncSandboxClient.fork()` is the same, awaited. `timeout_seconds` bounds the
checkpoint or pause wait, and the call returns as soon as the create is accepted.
A memory fork's pause and resume are asynchronous: the source stays PAUSED until
the platform reports it RUNNING, and if the fork fails after the pause, the error
names the source (or the created fork) so either can be recovered or deleted.
If the pause request itself fails or its response is lost, the source may still
become PAUSED; resume it with `resume()` (`prime sandbox resume <id>`).
A memory fork is an exact copy of the paused guest: the kernel RNG is reseeded,
but userspace random generators (OpenSSL, language runtimes) and keys, tokens,
or IDs already in memory are the same in every fork, so rekey or regenerate them
in the workload. GPU sandboxes and sandboxes with volumes cannot be paused, so
they cannot be memory-forked; use the default disk fork for those.

### Memory Snapshots

`snapshot()` captures the source's memory and running processes without creating
another sandbox. `checkpoint()` stays filesystem-only:

```python
snapshot = sandbox_client.snapshot(sandbox.id)
# The source is resuming asynchronously; the snapshot may still be uploading.
restored = sandbox_client.create(CreateSandboxRequest(
    name="restored-sandbox",
    checkpoint_id=snapshot.id,
    restore_memory=True,
))
```

`AsyncSandboxClient.snapshot()` is the same, awaited. The call pauses the source,
waits until the pause checkpoint is `FROZEN` or `DURABLE` and the source is
`PAUSED`, then requests a resume of the same source and returns the checkpoint it
observed. The resume is accepted asynchronously: the source reports `RUNNING`
later, and `timeout_seconds` (default 300) bounds only the pause-ready wait. The
returned checkpoint is not necessarily `DURABLE`; `wait_for_checkpoint` waits for
disk durability, not proof that the memory snapshot was stored. A non-durable
snapshot is usable only on the pausing node and is discarded if the source is
deleted before its upload completes, so do not delete the source before the
checkpoint is `DURABLE`; after `DURABLE` it survives source deletion. Paused
time counts against the source's timeout, and a pause checkpoint that does not
become `DURABLE` within the
platform's pause window (currently 6 hours) becomes `FAILED`. A `DURABLE` pause
whose memory misses that same window stays `DURABLE` and loses its memory claim.
If the source does not resume after the capture, the raised error names the
checkpoint id and the source id. If the pause request fails or its response is
lost, the source may still become PAUSED; check `get()` and continue it with
`resume()` (`prime sandbox resume <id>`). The result is a `SandboxCheckpoint`,
so `checkpoint restore --memory` and other memory restores take its id.

No sandbox is created. Restoring always creates a new sandbox; the platform does
not restore into the same running sandbox. A memory restore needs the source
sandbox record to exist: the platform keeps it about 7 days after termination,
then archives it, and disk-only restore is the remaining option. A pause
checkpoint has no TTL. GPU sandboxes and sandboxes with volumes cannot be paused.
A memory snapshot copies userspace random state and keys or tokens already in the
guest's memory; rekey or regenerate them in a workload restored from it.

### Reattach a Live Process

A live process keeps running when its handle goes away: the client that started it
can exit, and the sandbox can be paused and resumed. Another client attaches to it
by its `session_uuid`:

```python
process = await client.open_process(sandbox.id, "python3 -u worker.py")
session = process.session_uuid  # store it somewhere the next client can read
await process.detach()  # release the stream; unlike aclose(), it sends no signal

# Later, from any client:
process = await client.attach_process(sandbox.id, session)
await process.write_stdin(b"next\n")
```

A sandbox created from a memory snapshot continues the processes of the sandbox
that was paused. Attach to one with the new sandbox's id and the same `session_uuid`.

With output replay, the attached handle gets the output that sandboxd retains from
the start of the process. An older sandboxd sends only the output from the attach on.
A process that exited within sandboxd's retention window reports its exit code.

### File Operations

```python
# Upload a file
sandbox_client.upload_file(
    sandbox_id=sandbox.id,
    file_path="/app/script.py",
    local_file_path="./local_script.py"
)

# Download a file
sandbox_client.download_file(
    sandbox_id=sandbox.id,
    file_path="/app/output.txt",
    local_file_path="./output.txt"
)
```

### Bulk Operations

```python
# Create multiple sandboxes
sandbox_ids = []
for i in range(5):
    sandbox = sandbox_client.create(CreateSandboxRequest(
        name=f"sandbox-{i}",
        docker_image="python:3.11-slim",
    ))
    sandbox_ids.append(sandbox.id)

# Wait for up to 100 sandboxes with one batched lifecycle-status request per poll
statuses = sandbox_client.bulk_wait_for_creation(sandbox_ids)

# Delete by IDs or labels
sandbox_client.bulk_delete(sandbox_ids=sandbox_ids)
# OR by labels
sandbox_client.bulk_delete(labels=["experiment-1"])
```

### Labels & Filtering

```python
# Create with labels
sandbox = sandbox_client.create(CreateSandboxRequest(
    name="labeled-sandbox",
    docker_image="python:3.11-slim",
    labels=["experiment", "ml-training"],
))

# List with filters
sandboxes = sandbox_client.list(
    status="RUNNING",
    labels=["experiment"],
    per_page=50,
)

for s in sandboxes.sandboxes:
    print(f"{s.name}: {s.status}")
```

Lists are cursor-paginated. When `next_cursor` is set, pass it as
`list(cursor=...)` to fetch the next page. Cursor pages return
`total=None`; only the first page (no `cursor`) includes `total`.

```python
response = sandbox_client.list(per_page=50)
while True:
    for s in response.sandboxes:
        print(f"{s.name}: {s.status}")
    if not response.next_cursor:
        break
    response = sandbox_client.list(per_page=50, cursor=response.next_cursor)
```

### Long-Running Tasks

Use `start_background_job` to run long-running tasks that continue after the API call returns. Poll for completion with `get_background_job`.

```python
from prime_sandboxes import APIClient, SandboxClient, CreateSandboxRequest

sandbox_client = SandboxClient(APIClient())

# Create sandbox with extended timeout
sandbox = sandbox_client.create(CreateSandboxRequest(
    name="training-job",
    docker_image="python:3.11-slim",
    timeout_minutes=1440,  # 24 hours
    cpu_cores=4,
    memory_gb=16,
))
sandbox_client.wait_for_creation(sandbox.id)

# Start a long-running job in the background
job = sandbox_client.start_background_job(
    sandbox.id,
    "python train.py --epochs 100"
)
print(f"Job started: {job.job_id}")

# VM sandboxes can check up to 100 SDK-started jobs across sandboxes with one
# platform request. Results preserve input order; completed jobs include the
# same bounded stdout/stderr tails as get_background_job().
statuses = sandbox_client.get_background_jobs([job])

# For latency-sensitive polling, status-only methods never download output.
# Fetch the hydrated result with get_background_job() after completion.
snapshots = sandbox_client.get_background_job_statuses([job])

# Poll for completion
import time
while True:
    status = sandbox_client.get_background_job(sandbox.id, job)
    if status.completed:
        print(f"Job finished with exit code: {status.exit_code}")
        print(status.stdout)
        break
    print("Still running...")
    time.sleep(30)

# Download results
sandbox_client.download_file(sandbox.id, "/app/model.pt", "./model.pt")
```

Status lookups are batched; gateways without the batch endpoint fall back to
per-job polling. Once an exit code is observed, completion
remains authoritative even if output retrieval exhausts its bounded retry
deadline: the unavailable stream is `None` and its `stdout_error` or
`stderr_error` field describes the retrieval failure.

Output downloads are deduplicated, cached within a bounded client-local LRU,
and scheduled separately from completion polling. Advanced callers can tune the
client-wide limits with `background_job_output_concurrency`,
`background_job_output_queue_size`, and `background_job_output_cache_bytes`;
the defaults are 20 active jobs, 200 queued jobs, and 64 MiB of cached streams.

#### Async version

```python
import asyncio
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

async def run_training():
    async with AsyncSandboxClient() as client:
        sandbox = await client.create(CreateSandboxRequest(
            name="async-training",
            docker_image="python:3.11-slim",
            timeout_minutes=720,
        ))
        await client.wait_for_creation(sandbox.id)

        # Start background job
        job = await client.start_background_job(
            sandbox.id,
            "python train.py"
        )

        # Poll until done
        while True:
            status = await client.get_background_job(sandbox.id, job)
            if status.completed:
                print(status.stdout)
                break
            await asyncio.sleep(30)

        await client.delete(sandbox.id)

asyncio.run(run_training())
```

## Documentation

Full API reference: https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-sandboxes

## Related Packages

- **`prime`** - Full CLI + SDK with pods, inference, and more (includes this package)

## License

MIT License - see LICENSE file for details
