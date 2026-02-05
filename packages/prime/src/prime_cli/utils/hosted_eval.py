import asyncio
import re
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from prime_cli.core import APIError, AsyncAPIClient
from prime_cli.core.config import Config
from prime_cli.utils.schemas import (
    EvalStatus,
    HostedEvalConfig,
    HostedEvalCreationResult,
    HostedEvalResult,
)

console = Console()

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
PROGRESS_BAR = re.compile(r".*\|[█▏▎▍▌▋▊▉ ]{10,}\|.*")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def filter_progress_bars(text: str) -> str:
    lines = text.splitlines()
    filtered = []
    for line in lines:
        if PROGRESS_BAR.search(line) or re.search(r"\d+%\|", line):
            if "100%" in line:
                match = re.search(r"([^|]*100%\|[█▏▎▍▌▋▊▉ ]+\|[^\n]*?)(?=\d+%\||$)", line)
                if match:
                    filtered.append(match.group(1).strip())
                else:
                    filtered.append(line)
            continue
        if line.strip():
            filtered.append(line)
    return "\n".join(filtered)


STATUS_MESSAGES = {
    "Waiting for container to start...",
    "No logs available",
    "Unable to retrieve logs",
    "Failed to fetch logs from sandbox",
}


def is_status_message(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(msg) for msg in STATUS_MESSAGES)


def clean_logs(text: str) -> str:
    cleaned = filter_progress_bars(strip_ansi(text))
    if is_status_message(cleaned):
        return ""
    return cleaned


def get_new_log_lines(old_logs: str, new_logs: str) -> list[str]:
    old_lines = old_logs.splitlines() if old_logs else []
    new_lines = new_logs.splitlines()

    if not old_logs:
        return new_lines

    overlap = 0
    max_overlap = min(len(old_lines), len(new_lines))
    for i in range(1, max_overlap + 1):
        if old_lines[-i:] == new_lines[:i]:
            overlap = i
    return new_lines[overlap:]


def parse_eval_status(status_str: str) -> EvalStatus | None:
    try:
        return EvalStatus(status_str)
    except ValueError:
        return None


async def create_hosted_evaluation(
    client: AsyncAPIClient,
    config: HostedEvalConfig,
) -> dict[str, Any]:
    eval_config: dict[str, Any] = {
        "num_examples": config.num_examples,
        "rollouts_per_example": config.rollouts_per_example,
        "allow_sandbox_access": config.allow_sandbox_access,
        "allow_instances_access": config.allow_instances_access,
    }

    if config.env_args:
        eval_config["env_args"] = config.env_args

    if config.timeout_minutes:
        eval_config["timeout_minutes"] = config.timeout_minutes

    if config.custom_secrets:
        eval_config["custom_secrets"] = config.custom_secrets

    payload: dict[str, Any] = {
        "environment_ids": [config.environment_id],
        "inference_model": config.inference_model,
        "eval_config": eval_config,
    }

    if config.name:
        payload["name"] = config.name

    return await client.post("/hosted-evaluations", json=payload)


async def get_evaluation(client: AsyncAPIClient, evaluation_id: str) -> dict[str, Any]:
    return await client.get(f"/evaluations/{evaluation_id}")


async def get_evaluation_logs(client: AsyncAPIClient, evaluation_id: str) -> str:
    try:
        response = await client.get(f"/hosted-evaluations/{evaluation_id}/logs")
        return response.get("logs") or ""
    except APIError:
        return ""


async def create_and_return_hosted_evaluation(
    config: HostedEvalConfig,
) -> HostedEvalCreationResult:
    async with AsyncAPIClient() as client:
        console.print(
            f"[cyan]Creating hosted evaluation for environment {config.environment_id}[/cyan]"
        )
        console.print(f"[dim]Model: {config.inference_model}[/dim]")
        console.print(
            f"[dim]Configuration: num_examples={config.num_examples}, "
            f"rollouts_per_example={config.rollouts_per_example}[/dim]"
        )
        console.print()

        result = await create_hosted_evaluation(client, config)
        evaluation_id = result.get("evaluation_id")

        if not evaluation_id:
            raise APIError(f"Failed to get evaluation ID from response. Response: {result}")

        return HostedEvalCreationResult(
            evaluation_id=evaluation_id,
            viewer_url=result.get("viewer_url"),
        )


async def follow_hosted_evaluation(
    evaluation_id: str,
    poll_interval: float = 10.0,
    stream_logs: bool = True,
) -> HostedEvalResult:
    async with AsyncAPIClient() as client:
        last_logs = ""
        consecutive_errors = 0

        with Live(
            Panel(
                Text.assemble(
                    "[cyan]⠋[/cyan]",
                    " Waiting for evaluation to start...",
                ),
                title="[bold]Hosted Evaluation[/bold]",
                border_style="blue",
            ),
            refresh_per_second=4,
            console=console,
        ) as live:
            first_poll = True
            while True:
                if not first_poll:
                    await asyncio.sleep(poll_interval)
                first_poll = False

                try:
                    eval_data = await get_evaluation(client, evaluation_id)
                    status_str = eval_data.get("status", "UNKNOWN")
                    status = parse_eval_status(status_str)
                    consecutive_errors = 0

                    status_color = status.color if status else "white"
                    total_samples = eval_data.get("total_samples", 0)
                    status_text = Text.assemble(
                        "Status: ",
                        (status_str, status_color),
                        f" | Samples: {total_samples}",
                    )
                    live.update(
                        Panel(
                            status_text,
                            title="[bold]Hosted Evaluation[/bold]",
                            border_style="blue",
                        )
                    )

                    if stream_logs and status in EvalStatus.has_logs_statuses():
                        raw_logs = await get_evaluation_logs(client, evaluation_id)
                        logs = clean_logs(raw_logs) if raw_logs else ""

                        if logs and logs != last_logs:
                            for line in get_new_log_lines(last_logs, logs):
                                live.console.print(line)
                            last_logs = logs

                    if status in EvalStatus.terminal_statuses():
                        live.stop()
                        break

                except APIError as e:
                    consecutive_errors += 1
                    if "429" in str(e):
                        if consecutive_errors >= 3:
                            live.console.print("[yellow]Rate limited. Waiting 30s...[/yellow]")
                            await asyncio.sleep(30)
                        else:
                            await asyncio.sleep(10)
                        continue
                    raise

        eval_data = await get_evaluation(client, evaluation_id)
        final_logs = await get_evaluation_logs(client, evaluation_id)
        final_status = parse_eval_status(eval_data.get("status", "")) or EvalStatus.FAILED

        return HostedEvalResult(
            evaluation_id=evaluation_id,
            status=final_status,
            viewer_url=eval_data.get("viewer_url"),
            total_samples=eval_data.get("total_samples", 0),
            avg_score=eval_data.get("avg_score"),
            min_score=eval_data.get("min_score"),
            max_score=eval_data.get("max_score"),
            error_message=eval_data.get("error_message"),
            logs=final_logs,
        )


async def run_hosted_evaluation(
    config: HostedEvalConfig,
    poll_interval: float = 10.0,
    stream_logs: bool = True,
    follow: bool = True,
) -> HostedEvalResult:
    creation_result = await create_and_return_hosted_evaluation(config)
    evaluation_id = creation_result.evaluation_id

    console.print(f"[green]✓ Created hosted evaluation:[/green] {evaluation_id}")
    console.print()

    if not follow:
        return HostedEvalResult(
            evaluation_id=evaluation_id,
            status=EvalStatus.PENDING,
            viewer_url=creation_result.viewer_url,
            total_samples=0,
            avg_score=None,
            min_score=None,
            max_score=None,
            error_message=None,
            logs=None,
        )

    return await follow_hosted_evaluation(
        evaluation_id=evaluation_id,
        poll_interval=poll_interval,
        stream_logs=stream_logs,
    )


def print_hosted_result(result: HostedEvalResult) -> None:
    console.print()
    console.rule("[bold]Hosted Evaluation Results[/bold]")
    console.print()

    # Show clear success/failure indicator
    if result.status == EvalStatus.COMPLETED:
        console.print("[bold green]✓ Evaluation completed successfully![/bold green]")
    elif result.status == EvalStatus.FAILED:
        console.print("[bold red]✗ Evaluation failed[/bold red]")
    elif result.status == EvalStatus.CANCELLED:
        console.print("[bold yellow]○ Evaluation was cancelled[/bold yellow]")

    console.print()
    console.print(f"[cyan]Evaluation ID:[/cyan] {result.evaluation_id}")

    status_color = result.status.color
    console.print(f"[cyan]Status:[/cyan] [{status_color}]{result.status.value}[/{status_color}]")
    console.print(f"[cyan]Total samples:[/cyan] {result.total_samples}")

    if result.avg_score is not None:
        console.print(f"[cyan]Average score:[/cyan] {result.avg_score:.4f}")
    if result.min_score is not None:
        console.print(f"[cyan]Min score:[/cyan] {result.min_score:.4f}")
    if result.max_score is not None:
        console.print(f"[cyan]Max score:[/cyan] {result.max_score:.4f}")

    console.print()

    viewer_url = result.viewer_url
    if not viewer_url:
        # Fallback: construct URL from frontend_url and eval_id
        config = Config()
        viewer_url = f"{config.frontend_url}/dashboard/rft/evals/{result.evaluation_id}"
    console.print(f"[bold green]View results:[/bold green] {viewer_url}")

    if result.error_message:
        console.print(f"\n[red]Error:[/red] {result.error_message}")

    console.print()
    console.print("[dim]CLI commands:[/dim]")
    console.print(f"  prime eval get {result.evaluation_id}")
    console.print(f"  prime eval samples {result.evaluation_id}")
    console.print(f"  prime eval logs {result.evaluation_id}")
    console.print()


def stop_hosted_evaluation(eval_id: str) -> None:
    try:

        async def do_stop():
            async with AsyncAPIClient() as client:
                return await client.post(f"/hosted-evaluations/{eval_id}/cancel")

        result = asyncio.run(do_stop())
        status = result.get("status", "CANCELLED")
        message = result.get("message", "Evaluation cancelled successfully")
        console.print(f"[green]✓ {message}[/green]")
        console.print(f"[dim]Status: {status}[/dim]")
        console.print()
        console.print(
            "[dim]The sandbox has been terminated. "
            "Any partial results may still be available.[/dim]"
        )
        console.print(f"[dim]View results: prime eval get {eval_id}[/dim]")
    except APIError as e:
        console.print(f"[red]Error stopping evaluation:[/red] {e}")
        raise SystemExit(1)


def extend_hosted_evaluation_timeout(eval_id: str, additional_minutes: int) -> None:
    """Extend the timeout of a running hosted evaluation."""
    try:

        async def do_extend():
            async with AsyncAPIClient() as client:
                return await client.patch(
                    f"/hosted-evaluations/{eval_id}/timeout",
                    json={"additional_minutes": additional_minutes},
                )

        result = asyncio.run(do_extend())
        new_timeout = result.get("new_timeout_minutes", "unknown")
        console.print(
            f"[green]✓ Extended timeout for evaluation {eval_id}[/green]\n"
            f"[dim]New total timeout: {new_timeout} minutes[/dim]"
        )
    except APIError as e:
        console.print(f"[red]Error extending timeout:[/red] {e}")
        raise SystemExit(1)
