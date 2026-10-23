# Slurm Cluster Resize

This document describes the `prime factory slurm resize` command for resizing Slurm clusters in the Model Factory.

## Overview

The `prime factory slurm resize` command allows you to change the node count of a Slurm cluster deployment. This feature enables you to scale your cluster up or down based on your workload requirements.

**Requirements:**
- Team admin privileges
- Cluster must have self-service enabled
- Cluster must be in running state

**API Contract:** Based on [platform PR #6917](https://github.com/PrimeIntellect-ai/platform/pull/6917)

## Usage

```bash
prime factory slurm resize <cluster_id> --nodes <count> [options]
```

### Arguments

- `cluster_id`: Cluster/job ID (the `id` field from `prime factory slurm list`)

### Options

- `--nodes, -n`: **Required.** New total node count (absolute target, not delta; must be >= 1 and <= max(64, current size))
- `--yes, -y`: Skip confirmation prompt
- `--wait, -w`: Wait until the resize operation completes (polls status every 5 seconds, times out after 5 minutes)
- `--team, -t`: Team ID override (defaults to the selected account context)
- `--json`: Print the API response as JSON
- `--output, -o`: Output format: table or json

## Examples

### List Slurm Clusters

First, get the cluster ID:

```bash
prime factory slurm list
```

Output:
```
┌─────────┬────────────────────┬─────────┬──────────┬──────┐
│ ID      │ NAME               │ STATUS  │ GPU TYPE │ GPUS │
├─────────┼────────────────────┼─────────┼──────────┼──────┤
│ job-123 │ research-cluster   │ running │ H100     │   32 │
└─────────┴────────────────────┴─────────┴──────────┴──────┘
```

### Basic Resize with Confirmation

Resize cluster to 4 nodes with confirmation:

```bash
prime factory slurm resize job-123 --nodes 4
```

Output:
```
Resize Slurm cluster 'research-cluster':
  Cluster ID:   job-123
  Current size: 32 GPUs
  New nodes:    4 nodes

Proceed with resize? [y/N]: y
Resize initiated for cluster 'research-cluster'
Target nodes: 4
Nodes to add: node-3, node-4
```

### Resize Without Confirmation

Use `--yes` to skip the confirmation prompt:

```bash
prime factory slurm resize job-123 --nodes 4 --yes
```

### Resize and Wait for Completion

Use `--wait` to poll the status until the resize completes:

```bash
prime factory slurm resize job-123 --nodes 4 --yes --wait
```

Output:
```
Resize initiated for cluster 'research-cluster'
Target nodes: 4
Nodes to add: node-3, node-4

Waiting for resize to complete (reconciliation is asynchronous)...
  Status: running | GPUs: 32
✓ Cluster is running
```

### JSON Output

Get machine-readable output:

```bash
prime factory slurm resize job-123 --nodes 4 --yes --json
```

Output:
```json
{
  "numNodes": 4,
  "added": ["node-3", "node-4"],
  "removed": []
}
```

### Scale Down

You can also scale down a cluster by specifying fewer nodes:

```bash
prime factory slurm resize job-123 --nodes 2 --yes
```

Output:
```
Resize initiated for cluster 'research-cluster'
Target nodes: 2
Nodes to remove: node-3, node-4
```

## Workflow Integration

### Check Current Status

```bash
prime factory slurm list
prime factory slurm get job-123
```

### Resize the Cluster

```bash
prime factory slurm resize job-123 --nodes 4 --yes --wait
```

### Verify the New Configuration

```bash
prime factory slurm list
```

## Error Handling

The command provides clear error messages for common issues with helpful context:

### 400: Invalid Request or Insufficient Resources

```
Error: HTTP 400: Invalid numNodes
Ensure numNodes is >= 1 and <= max(64, current size) with sufficient resources.
```

### 403: Permission Denied

```
Error: HTTP 403: Forbidden
This operation requires team admin privileges and the cluster must have self-service enabled.
```

**Solution:** Verify you are a team admin and that the cluster has self-service enabled.

### 404: Cluster Not Found

```
Error: HTTP 404: Cluster not found
Cluster not found. Check the cluster ID and team context.
```

**Solution:** Use `prime factory slurm list` to get the correct cluster ID.

### 409: Concurrent Operation or Insufficient Idle Nodes

```
Error: HTTP 409: Conflict
The cluster may not be running, another operation is in progress, or there are insufficient idle nodes.
```

**Solution:** Wait for any ongoing operations to complete, or check cluster status.

### 502: Verification Failed

```
Error: HTTP 502: Bad Gateway
Node labels were changed but verification failed. Check cluster status.
```

**Solution:** Check cluster status and contact support if the issue persists.

### 503: Slurm State Unavailable (Retryable)

```
Error: HTTP 503: Service Unavailable
Slurm state unavailable. This is likely temporary - please retry in a moment.
```

**Solution:** This is a temporary issue. Wait a moment and retry the command.

### No Team Selected

```
No team selected in the current account context.
`prime factory slurm` manages your team's Slurm deployments.
Run `prime switch` to select a team, or pass --team <team_id>.
```

**Solution:** Run `prime switch` to select a team context.

## Implementation Details

### Backend API Contract

The CLI calls the following endpoint (from [platform PR #6917](https://github.com/PrimeIntellect-ai/platform/pull/6917)):

- **Method:** `POST`
- **Endpoint:** `/api/v1/slurm-clusters/{team_id}/{job_id}/resize`
- **Authentication:** Bearer token (API key) with `slurm_clusters:write` scope
- **Authorization:** Caller must be team admin; cluster must have `selfServiceEnabled=true`
- **Request Body:**
  ```json
  {
    "numNodes": 4
  }
  ```
- **200 Response:**
  ```json
  {
    "numNodes": 4,
    "added": ["node-3", "node-4"],
    "removed": []
  }
  ```

### Constraints

- `numNodes` must be >= 1 and <= max(64, current_size)
- The target is an absolute node count, not a delta
- Resize is asynchronous - the reconciler adds/removes nodes after the API returns
- Only one resize operation can be in progress at a time per cluster

### Status Polling

When `--wait` is used:
- Polls `prime factory slurm list` every 5 seconds
- Checks cluster status (running, ready, online indicate completion)
- Times out after 5 minutes (300 seconds)
- Shows current status and GPU count on each poll

### Node Count vs. GPU Count

The API works with **node counts**, but `prime factory slurm list` shows **GPU counts**. The confirmation prompt displays the GPU count from the current cluster state and the target node count from your `--nodes` argument. After resize, verify the result by checking the updated GPU count in the cluster list.

## See Also

- `prime factory slurm list` - List your team's Slurm deployments
- `prime factory slurm get <id>` - Show one deployment's details
- `prime factory slurm members <id>` - List members with SSH access
- `prime factory status` - View overall factory cluster allocation
- `prime switch` - Switch between personal and team contexts
