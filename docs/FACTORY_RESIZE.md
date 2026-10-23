# Model Factory Cluster Resize

This document describes the `prime factory resize` command for resizing Model Factory clusters.

## Overview

The `prime factory resize` command allows you to change the total GPU count of a dedicated Model Factory cluster. This feature enables you to scale your cluster up or down based on your workload requirements.

## Usage

```bash
prime factory resize <cluster> --nodes <count> [options]
```

### Arguments

- `cluster`: Cluster display name or 1-based index (from `prime factory status`)

### Options

- `--nodes, -n`: **Required.** New total GPU count for the cluster
- `--yes, -y`: Skip confirmation prompt
- `--wait, -w`: Wait until the resize operation completes (polls status every 5 seconds, times out after 5 minutes)
- `--team, -t`: Team ID override (defaults to the selected account context)
- `--json`: Print the API response as JSON
- `--output, -o`: Output format: table or json

## Examples

### Basic Resize with Confirmation

Resize a cluster named "research-b300" to 256 GPUs with confirmation:

```bash
prime factory resize research-b300 --nodes 256
```

Output:
```
Resize cluster 'research-b300':
  Current size: 128 GPUs
  New size:     256 GPUs
  Change:       +128 GPUs (scale up)

Proceed with resize? [y/N]: y
Resize initiated for cluster 'research-b300'
Target size: 256 GPUs
```

### Resize Without Confirmation

Use `--yes` to skip the confirmation prompt:

```bash
prime factory resize research-b300 --nodes 256 --yes
```

### Resize by Index

If you have multiple clusters, you can use the 1-based index from `prime factory status`:

```bash
prime factory resize 1 --nodes 256 --yes
```

### Resize and Wait for Completion

Use `--wait` to poll the status until the resize completes:

```bash
prime factory resize research-b300 --nodes 256 --yes --wait
```

Output:
```
Resize initiated for cluster 'research-b300'
Target size: 256 GPUs

Waiting for resize to complete...
  Current: 128 GPUs (target: 256)
  Current: 192 GPUs (target: 256)
✓ Resize complete: 256 GPUs
```

### JSON Output

Get machine-readable output:

```bash
prime factory resize research-b300 --nodes 256 --yes --json
```

Output:
```json
{
  "status": "resize_initiated",
  "cluster_id": "research-b300",
  "target_gpus": 256
}
```

### Scale Down

You can also scale down a cluster:

```bash
prime factory resize research-b300 --nodes 64 --yes
```

Output:
```
Resize cluster 'research-b300':
  Current size: 128 GPUs
  New size:     64 GPUs
  Change:       -64 GPUs (scale down)

Proceed with resize? [y/N]:
```

## Workflow Integration

### Check Current Status

Before resizing, check the current cluster configuration:

```bash
prime factory status --cluster research-b300
```

### Resize the Cluster

```bash
prime factory resize research-b300 --nodes 256 --yes --wait
```

### Verify the New Configuration

```bash
prime factory status --cluster research-b300
```

## Error Handling

The command provides clear error messages for common issues:

### No Team Selected

```
No team selected in the current account context.
`prime factory resize` operates on your team's dedicated clusters.
Run `prime switch` to select a team, or pass --team <team_id>.
```

### Cluster Not Found

```
Error: No cluster matched 'nonexistent-cluster'.
Available clusters: [1] research-b300, [2] training-a100
```

### Invalid Node Count

```
Error: --nodes must be a positive integer (got -10).
```

### API Errors

```
Error: Insufficient capacity for resize
```

## Implementation Details

### Backend API Contract

The CLI calls the following endpoint:

- **Method:** `PATCH`
- **Endpoint:** `/api/v1/factory/clusters/{cluster_id}/resize`
- **Request Body:**
  ```json
  {
    "team_id": "team-123",
    "total_gpus": 256
  }
  ```
- **Response:** JSON object with resize status (structure depends on backend implementation)

The `cluster_id` in the endpoint path is currently the cluster's `display_name`. If the backend uses a different identifier, the SDK method should be updated accordingly.

### Status Polling

When `--wait` is used:
- Polls `prime factory status` every 5 seconds
- Compares `total_gpus` field with target
- Times out after 5 minutes (300 seconds)
- Shows current progress on each poll

## See Also

- `prime factory status` - View cluster allocation and GPU usage
- `prime factory nodes` - View individual node status
- `prime factory workloads` - View running workloads
- `prime switch` - Switch between personal and team contexts
